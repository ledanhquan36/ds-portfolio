"""
End-to-end churn pipeline: EDA figures -> model selection -> calibration ->
profit-optimal threshold -> explainability -> reports/metrics.json

Run:  python 01_churn_prediction/src/train.py
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

import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.viz import PALETTE, save, set_style
from features import add_features, split_columns
from make_dataset import clean, load

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "reports" / "figures"
SEED = 42

# --- business assumptions for the retention campaign (documented, not hidden) ---
OFFER_COST = 60.0       # USD, one-off retention incentive per contacted customer
ACCEPT_RATE = 0.30      # share of true churners the offer actually saves
MARGIN = 0.35           # gross margin on revenue
HORIZON_MONTHS = 12     # value horizon of a saved customer


def build_preprocessor(num, cat, scale: bool):
    num_steps = [("scale", StandardScaler())] if scale else [("pass", "passthrough")]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), cat),
        ]
    )


def candidate_models(num, cat):
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(num, cat, scale=True)),
                ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                           random_state=SEED)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(num, cat, scale=False)),
                ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=8,
                                               max_features="sqrt",
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=SEED)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor(num, cat, scale=False)),
                ("clf", HistGradientBoostingClassifier(learning_rate=0.06, max_iter=400,
                                                       max_leaf_nodes=15, min_samples_leaf=25,
                                                       l2_regularization=1.0,
                                                       early_stopping=True,
                                                       validation_fraction=0.15,
                                                       random_state=SEED)),
            ]
        ),
    }


# ----------------------------------------------------------------- EDA figures
def eda(df: pd.DataFrame) -> dict:
    set_style()
    rate = df.churn.mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        g = df.groupby(col).churn.agg(["mean", "size"]).sort_values("mean")
        ax.barh(g.index.astype(str), g["mean"] * 100, color=PALETTE[0])
        ax.axvline(rate * 100, color=PALETTE[4], ls="--", lw=1.2, label="base %.1f%%" % (rate * 100))
        ax.set_xlabel("churn rate (%)")
        ax.set_title(col)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Churn rate by contract, product and payment method", y=1.03, fontweight="bold")
    save(fig, FIG / "01_churn_by_segment.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    order = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
    g = df.groupby("tenure_bucket").churn.mean().reindex(order)
    axes[0].plot(g.index, g.values * 100, marker="o", color=PALETTE[0])
    axes[0].set_title("Churn rate by tenure bucket")
    axes[0].set_ylabel("churn rate (%)")
    for label, sub in df.groupby(df.churn.map({0: "stayed", 1: "churned"})):
        axes[1].hist(sub.MonthlyCharges, bins=30, alpha=0.6, label=label, density=True)
    axes[1].set_title("Monthly charges distribution")
    axes[1].set_xlabel("monthly charges (USD)")
    axes[1].legend()
    save(fig, FIG / "02_tenure_and_price.png")

    return {
        "n_rows": int(len(df)),
        "churn_rate": round(float(rate), 4),
        "churn_rate_by_contract_pct": {k: round(float(v) * 100, 1)
                                       for k, v in df.groupby("Contract").churn.mean().items()},
        "churn_rate_by_internet_pct": {k: round(float(v) * 100, 1)
                                       for k, v in df.groupby("InternetService").churn.mean().items()},
    }


# ------------------------------------------------------------- business metrics
def campaign_value(y_true, proba, monthly, threshold: float) -> float:
    """Net USD of contacting every customer scored at or above `threshold`."""
    contacted = proba >= threshold
    value_if_saved = monthly * HORIZON_MONTHS * MARGIN * ACCEPT_RATE
    gain = np.where(y_true == 1, value_if_saved, 0.0) - OFFER_COST
    return float(gain[contacted].sum())


def main() -> None:
    t0 = time.time()
    print("[1/6] data")
    df = add_features(clean(load()))
    stats = eda(df)
    print("  %d customers, churn rate %.1f%%" % (len(df), stats["churn_rate"] * 100))

    num, cat = split_columns(df)
    X = df[num + cat]
    y = df.churn.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    print("[2/6] model selection (5-fold stratified CV on the training split)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, pipe in candidate_models(num, cat).items():
        res = cross_validate(pipe, X_tr, y_tr, cv=cv,
                             scoring=["roc_auc", "average_precision"], n_jobs=-1)
        row = {
            "model": name,
            "roc_auc_mean": float(res["test_roc_auc"].mean()),
            "roc_auc_std": float(res["test_roc_auc"].std()),
            "pr_auc_mean": float(res["test_average_precision"].mean()),
        }
        cv_rows.append(row)
        print("  %-24s ROC-AUC %.4f +/- %.4f   PR-AUC %.4f"
              % (name, row["roc_auc_mean"], row["roc_auc_std"], row["pr_auc_mean"]))
    cv_table = pd.DataFrame(cv_rows).sort_values("roc_auc_mean", ascending=False)
    best_name = cv_table.iloc[0]["model"]
    print("  -> best: %s" % best_name)

    print("[3/6] fit + probability calibration (isotonic, 5-fold internal)")
    raw = candidate_models(num, cat)[best_name].fit(X_tr, y_tr)
    p_raw = raw.predict_proba(X_te)[:, 1]
    cal = CalibratedClassifierCV(candidate_models(num, cat)[best_name], method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]
    brier = {"uncalibrated": float(brier_score_loss(y_te, p_raw)),
             "calibrated": float(brier_score_loss(y_te, p_cal))}
    use_cal = brier["calibrated"] <= brier["uncalibrated"]
    final_model, proba = (cal, p_cal) if use_cal else (raw, p_raw)
    print("  Brier %.4f -> %.4f (using %s)"
          % (brier["uncalibrated"], brier["calibrated"], "calibrated" if use_cal else "uncalibrated"))

    print("[4/6] threshold tuned on expected campaign value")
    monthly_te = df.loc[X_te.index, "MonthlyCharges"].values
    grid = np.linspace(0.05, 0.95, 181)
    values = np.array([campaign_value(y_te, proba, monthly_te, t) for t in grid])
    f1s = np.array([f1_score(y_te, proba >= t) for t in grid])
    t_value, t_f1 = float(grid[values.argmax()]), float(grid[f1s.argmax()])
    pred = (proba >= t_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    order = np.argsort(-proba)
    deciles = np.array_split(order, 10)
    capture = [float(y_te[d].sum() / y_te.sum()) for d in deciles]
    lift_top_decile = float(y_te[deciles[0]].mean() / y_te.mean())

    metrics = {
        "dataset": stats,
        "cv_table": cv_rows,
        "best_model": best_name,
        "calibration_brier": brier,
        "test": {
            "n": int(len(y_te)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "threshold_profit_optimal": t_value,
            "threshold_f1_optimal": t_f1,
            "precision": float(precision_score(y_te, pred)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "lift_top_decile": lift_top_decile,
            "capture_at_top_30pct": float(sum(capture[:3])),
        },
        "business": {
            "assumptions": {
                "offer_cost_usd": OFFER_COST,
                "offer_accept_rate": ACCEPT_RATE,
                "gross_margin": MARGIN,
                "value_horizon_months": HORIZON_MONTHS,
            },
            "net_value_at_optimal_threshold_usd": float(values.max()),
            "net_value_contact_everyone_usd": campaign_value(y_te, proba, monthly_te, 0.0),
            "customers_contacted": int(pred.sum()),
            "per_1000_customers_usd": float(values.max() / len(y_te) * 1000),
        },
    }

    print("[5/6] evaluation figures + permutation importance")
    set_style()
    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(fpr, tpr, color=PALETTE[0], label="AUC = %.3f" % metrics["test"]["roc_auc"])
    axes[0].plot([0, 1], [0, 1], ls="--", color=PALETTE[5], lw=1)
    axes[0].set(title="ROC curve", xlabel="false positive rate", ylabel="true positive rate")
    axes[0].legend()
    axes[1].plot(rec, prec, color=PALETTE[1], label="PR-AUC = %.3f" % metrics["test"]["pr_auc"])
    axes[1].axhline(y_te.mean(), ls="--", color=PALETTE[5], lw=1, label="base rate")
    axes[1].set(title="Precision-Recall curve", xlabel="recall", ylabel="precision")
    axes[1].legend()
    for p_, lab in [(p_raw, "uncalibrated"), (p_cal, "isotonic calibrated")]:
        pt, pp = calibration_curve(y_te, p_, n_bins=10, strategy="quantile")
        axes[2].plot(pp, pt, marker="o", label=lab)
    axes[2].plot([0, 1], [0, 1], ls="--", color=PALETTE[5], lw=1)
    axes[2].set(title="Calibration (reliability)", xlabel="predicted probability",
                ylabel="observed churn rate")
    axes[2].legend(fontsize=8)
    save(fig, FIG / "03_model_evaluation.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grid, values, color=PALETTE[2])
    axes[0].axvline(t_value, color=PALETTE[4], ls="--", label="optimal t = %.2f" % t_value)
    axes[0].axhline(0, color=PALETTE[5], lw=1)
    axes[0].set(title="Expected campaign value vs threshold", xlabel="score threshold",
                ylabel="net value on test set (USD)")
    axes[0].legend()
    cum = np.cumsum(capture) * 100
    axes[1].plot(np.arange(1, 11) * 10, cum, marker="o", color=PALETTE[0], label="model")
    axes[1].plot([0, 100], [0, 100], ls="--", color=PALETTE[5], lw=1, label="random")
    axes[1].set(title="Cumulative gain", xlabel="% of base contacted (by score)",
                ylabel="% of churners captured")
    axes[1].legend()
    save(fig, FIG / "04_business_value.png")

    perm = permutation_importance(final_model, X_te, y_te, n_repeats=8, random_state=SEED,
                                  scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False).head(12)
    metrics["top_features_permutation_auc_drop"] = {k: round(float(v), 4) for k, v in imp.items()}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index[::-1], imp.values[::-1], color=PALETTE[0])
    ax.set(title="Permutation importance (drop in test ROC-AUC)", xlabel="AUC drop")
    save(fig, FIG / "05_feature_importance.png")

    print("[6/6] persist artefacts")
    joblib.dump(final_model, PROJ / "reports" / "churn_model.joblib")
    (PROJ / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cv_table.to_csv(PROJ / "reports" / "cv_results.csv", index=False)

    scored = df.loc[X_te.index, ["customerID", "Contract", "tenure", "MonthlyCharges"]].copy()
    scored["churn_probability"] = proba.round(4)
    scored["actual_churn"] = y_te
    scored["target_for_campaign"] = pred
    scored.sort_values("churn_probability", ascending=False).to_csv(
        PROJ / "reports" / "scored_customers.csv", index=False)

    net = metrics["business"]["net_value_at_optimal_threshold_usd"]
    print("done in %.1fs | test ROC-AUC %.3f | net value $%s on %d held-out customers"
          % (time.time() - t0, metrics["test"]["roc_auc"], format(round(net), ","), len(y_te)))


if __name__ == "__main__":
    main()
