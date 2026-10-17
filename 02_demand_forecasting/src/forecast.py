"""
Hourly demand forecasting, 24 hours ahead, on the UCI bike-sharing series.

Framing: at 00:00 we must publish an hourly staffing/rebalancing plan for the next
day, so a feature may only use information available 24h before the timestamp it
predicts. Every lag is therefore >= 24h and every rolling window is shifted by 24h.

Validation is strictly chronological: expanding-window CV for model choice, then a
held-out final 8 weeks that no model saw during selection. Baselines are the ones
an operations team would actually use (same hour yesterday / same hour last week).

Run:  python 02_demand_forecasting/src/forecast.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.viz import PALETTE, save, set_style
from make_dataset import load

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "reports" / "figures"
SEED = 42
HORIZON = 24          # hours ahead
TEST_WEEKS = 8
TARGET = "count"


# --------------------------------------------------------------- feature build
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regular hourly grid + calendar, weather and lag features."""
    s = df.set_index("timestamp").sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(grid)
    s.index.name = "timestamp"

    out = pd.DataFrame(index=s.index)
    out["y"] = s[TARGET]

    # calendar
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["year"] = out.index.year
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    out["dayofyear"] = out.index.dayofyear
    # cyclical encodings so the linear model can see daily/annual periodicity
    out["hour_sin"] = np.sin(2 * np.pi * out.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out.dayofyear / 365.25)

    # exogenous columns that a day-ahead forecast can legitimately use:
    # holiday/workingday are known from the calendar, weather from the forecast
    for col in ["holiday", "workingday", "weather", "season"]:
        out[col] = s[col].astype(object).astype(str).replace("nan", np.nan).ffill()
    for col in ["temp", "feel_temp", "humidity", "windspeed"]:
        out[col] = pd.to_numeric(s[col], errors="coerce").interpolate(limit_direction="both")

    # lags: only values at least HORIZON hours old
    for lag in [24, 25, 26, 48, 72, 168, 336]:
        out["lag_%dh" % lag] = out.y.shift(lag)
    base = out.y.shift(HORIZON)
    out["roll_mean_24h"] = base.rolling(24, min_periods=12).mean()
    out["roll_std_24h"] = base.rolling(24, min_periods=12).std()
    out["roll_mean_168h"] = base.rolling(168, min_periods=84).mean()
    out["roll_max_24h"] = base.rolling(24, min_periods=12).max()
    # same hour, averaged over the previous 3 weeks -> stable weekly profile
    out["same_hour_3w_mean"] = (
        pd.concat([out.y.shift(168), out.y.shift(336), out.y.shift(504)], axis=1).mean(axis=1)
    )
    out["trend"] = np.arange(len(out))
    return out


NUM = [
    "hour", "dayofweek", "month", "year", "is_weekend", "dayofyear",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temp", "feel_temp", "humidity", "windspeed",
    "lag_24h", "lag_25h", "lag_26h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
    "roll_mean_24h", "roll_std_24h", "roll_mean_168h", "roll_max_24h",
    "same_hour_3w_mean", "trend",
]
CAT = ["holiday", "workingday", "weather", "season"]


def models(cat_for_linear: bool = True) -> dict[str, Pipeline]:
    prep_linear = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ]
    )
    prep_tree = ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ]
    )
    return {
        "ridge": Pipeline([("prep", prep_linear), ("m", Ridge(alpha=3.0, random_state=SEED))]),
        "hgb_squared": Pipeline(
            [("prep", prep_tree),
             ("m", HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.06,
                                                 max_iter=600, max_leaf_nodes=31,
                                                 min_samples_leaf=20, l2_regularization=1.0,
                                                 early_stopping=True, validation_fraction=0.1,
                                                 random_state=SEED))]),
        "hgb_poisson": Pipeline(
            [("prep", prep_tree),
             ("m", HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06,
                                                 max_iter=600, max_leaf_nodes=31,
                                                 min_samples_leaf=20, l2_regularization=1.0,
                                                 early_stopping=True, validation_fraction=0.1,
                                                 random_state=SEED))]),
    }


def smape(y, p) -> float:
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(np.mean(np.abs(y - p) / np.where(denom == 0, 1, denom)) * 100)


def score(y, p) -> dict:
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "smape_pct": smape(y, p),
        "r2": float(r2_score(y, p)),
    }


def main() -> None:
    t0 = time.time()
    print("[1/6] data + features")
    raw = load()
    feat = build_features(raw)
    data = feat.dropna(subset=["y"] + [c for c in NUM if c.startswith(("lag_", "roll_", "same_"))])
    print("  usable rows %d of %d (lag warm-up and service gaps dropped)" % (len(data), len(feat)))

    cutoff = data.index.max() - pd.Timedelta(weeks=TEST_WEEKS)
    train, test = data[data.index <= cutoff], data[data.index > cutoff]
    X_cols = NUM + CAT
    X_tr, y_tr = train[X_cols], train.y.values
    X_te, y_te = test[X_cols], test.y.values
    print("  train %s .. %s (%d) | test %s .. %s (%d)"
          % (train.index.min().date(), train.index.max().date(), len(train),
             test.index.min().date(), test.index.max().date(), len(test)))

    print("[2/6] baselines an ops team would use")
    baselines = {
        "naive_same_hour_yesterday": test.lag_24h.values,
        "seasonal_naive_same_hour_last_week": test.lag_168h.values,
        "mean_same_hour_last_3_weeks": test.same_hour_3w_mean.values,
    }
    base_scores = {k: score(y_te, v) for k, v in baselines.items()}
    for k, v in base_scores.items():
        print("  %-36s MAE %6.1f  RMSE %6.1f  sMAPE %5.1f%%" % (k, v["mae"], v["rmse"], v["smape_pct"]))

    print("[3/6] expanding-window CV (5 folds, chronological)")
    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 14)
    cv_rows = []
    for name, pipe in models().items():
        maes = []
        for tr_idx, va_idx in tscv.split(X_tr):
            p = pipe.fit(X_tr.iloc[tr_idx], y_tr[tr_idx]).predict(X_tr.iloc[va_idx])
            maes.append(mean_absolute_error(y_tr[va_idx], np.clip(p, 0, None)))
        cv_rows.append({"model": name, "cv_mae_mean": float(np.mean(maes)),
                        "cv_mae_std": float(np.std(maes)), "fold_maes": [round(m, 2) for m in maes]})
        print("  %-12s CV MAE %6.2f +/- %.2f   folds %s"
              % (name, np.mean(maes), np.std(maes), [round(m, 1) for m in maes]))
    cv_table = pd.DataFrame(cv_rows).sort_values("cv_mae_mean")
    best_name = cv_table.iloc[0]["model"]
    print("  -> best: %s" % best_name)

    print("[4/6] refit on full training window, evaluate on held-out 8 weeks")
    best = models()[best_name].fit(X_tr, y_tr)
    pred = np.clip(best.predict(X_te), 0, None)
    test_scores = {name: score(y_te, np.clip(models()[name].fit(X_tr, y_tr).predict(X_te), 0, None))
                   for name in models()}
    for k, v in test_scores.items():
        print("  %-12s MAE %6.1f  RMSE %6.1f  sMAPE %5.1f%%  R2 %.3f"
              % (k, v["mae"], v["rmse"], v["smape_pct"], v["r2"]))

    print("[5/6] prediction intervals (quantile gradient boosting, 5%-95%)")
    q_models = {}
    for q in (0.05, 0.95):
        q_models[q] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.06, max_iter=400,
            max_leaf_nodes=31, min_samples_leaf=20, early_stopping=True,
            validation_fraction=0.1, random_state=SEED)
    prep = ColumnTransformer([("num", "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    Xtr_m = prep.fit_transform(X_tr)
    Xte_m = prep.transform(X_te)
    lo = np.clip(q_models[0.05].fit(Xtr_m, y_tr).predict(Xte_m), 0, None)
    hi = np.clip(q_models[0.95].fit(Xtr_m, y_tr).predict(Xte_m), 0, None)
    coverage = float(np.mean((y_te >= lo) & (y_te <= hi)))
    print("  nominal 90%% interval -> empirical coverage %.1f%%, mean width %.0f rides"
          % (coverage * 100, float(np.mean(hi - lo))))

    best_mae = test_scores[best_name]["mae"]
    bl_mae = base_scores["seasonal_naive_same_hour_last_week"]["mae"]
    metrics = {
        "task": "24h-ahead hourly demand forecast, UCI bike sharing 2011-2012",
        "rows_total": int(len(raw)),
        "rows_modelled": int(len(data)),
        "train_window": [str(train.index.min()), str(train.index.max())],
        "test_window": [str(test.index.min()), str(test.index.max())],
        "baselines_test": base_scores,
        "cv_table": cv_rows,
        "best_model": best_name,
        "models_test": test_scores,
        "improvement_vs_seasonal_naive_pct": round(float((bl_mae - best_mae) / bl_mae * 100), 1),
        "prediction_interval": {"nominal": 0.90, "empirical_coverage": coverage,
                                "mean_width_rides": float(np.mean(hi - lo))},
    }

    print("[6/6] figures")
    set_style()
    res = pd.DataFrame({"y": y_te, "pred": pred, "lo": lo, "hi": hi}, index=test.index)
    window = res.loc[res.index >= res.index.max() - pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(window.index, window.lo, window.hi, color=PALETTE[0], alpha=0.18,
                    label="90% prediction interval")
    ax.plot(window.index, window.y, color="#222", lw=1.4, label="actual")
    ax.plot(window.index, window.pred, color=PALETTE[1], lw=1.4, label="forecast (%s)" % best_name)
    ax.set(title="Day-ahead hourly demand, final two weeks of the held-out period",
           ylabel="rides per hour")
    ax.legend(ncol=3)
    save(fig, FIG / "01_forecast_vs_actual.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    err = res.assign(abs_err=(res.y - res.pred).abs(), hour=res.index.hour,
                     dow=res.index.dayofweek)
    axes[0].bar(err.groupby("hour").abs_err.mean().index,
                err.groupby("hour").abs_err.mean().values, color=PALETTE[0])
    axes[0].set(title="MAE by hour of day", xlabel="hour", ylabel="MAE (rides)")
    dow_mae = err.groupby("dow").abs_err.mean()
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow_mae.values, color=PALETTE[2])
    axes[1].set(title="MAE by day of week", ylabel="MAE (rides)")
    axes[2].hist(res.y - res.pred, bins=40, color=PALETTE[3])
    axes[2].axvline(0, color="#222", lw=1)
    axes[2].set(title="Residual distribution", xlabel="actual - forecast")
    save(fig, FIG / "02_error_analysis.png")

    profile = feat.dropna(subset=["y"]).assign(
        kind=np.where(feat.dropna(subset=["y"]).is_weekend == 1, "weekend", "weekday"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind, sub in profile.groupby("kind"):
        axes[0].plot(sub.groupby("hour").y.mean(), marker="o", label=kind)
    axes[0].set(title="Average hourly demand profile", xlabel="hour", ylabel="rides")
    axes[0].legend()
    monthly = feat.dropna(subset=["y"]).resample("MS").y.mean()
    axes[1].plot(monthly.index, monthly.values, marker="o", color=PALETTE[1])
    axes[1].set(title="Monthly average demand (growth + seasonality)", ylabel="rides per hour")
    save(fig, FIG / "03_demand_profile.png")

    names = list(models().keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["seasonal naive"] + names
    vals = [bl_mae] + [test_scores[n]["mae"] for n in names]
    ax.bar(labels, vals, color=[PALETTE[5]] + PALETTE[:3])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, "%.1f" % v, ha="center", fontsize=9)
    ax.set(title="Held-out MAE: baseline vs models", ylabel="MAE (rides per hour)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, FIG / "04_model_comparison.png")

    perm = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=SEED,
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_cols).sort_values(ascending=False).head(12)
    metrics["top_features_permutation_mae_increase"] = {k: round(float(v), 3) for k, v in imp.items()}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index[::-1], imp.values[::-1], color=PALETTE[0])
    ax.set(title="Permutation importance (increase in MAE)", xlabel="MAE increase (rides)")
    save(fig, FIG / "05_feature_importance.png")

    (PROJ / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    res.round(2).to_csv(PROJ / "reports" / "forecast_test.csv")
    cv_table.to_csv(PROJ / "reports" / "cv_results.csv", index=False)

    print("done in %.1fs | %s MAE %.1f vs seasonal-naive %.1f (%.1f%% better)"
          % (time.time() - t0, best_name, best_mae, bl_mae,
             metrics["improvement_vs_seasonal_naive_pct"]))


if __name__ == "__main__":
    main()
