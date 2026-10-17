"""
Experimentation (A/B test) analysis toolkit, demonstrated on a simulated
checkout experiment where the ground truth is known - that is the only way to
show that the machinery is correct.

What it covers, in the order an analyst actually works:
  1. design        - power curves and the minimum detectable effect for the traffic we have
  2. sanity        - sample-ratio-mismatch check before looking at any metric
  3. inference     - two-proportion z-test, Welch t-test, bootstrap CI, delta method
                     for the ratio metric (revenue per session)
  4. sensitivity   - CUPED variance reduction using a pre-experiment covariate
  5. peeking       - the false-positive cost of daily checks, and a simulation-calibrated
                     sequential boundary that fixes it
  6. many metrics  - Benjamini-Hochberg control across the metric suite
  7. decision      - Bayesian posterior, P(treatment wins), expected loss, and a
                     ship / no-ship call written against a pre-registered threshold

Run:  python 04_rfm_segmentation/src/ab_test_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from common.viz import PALETTE, save, set_style

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "reports" / "figures"
SEED = 7
rng = np.random.default_rng(SEED)

# --- experiment design, fixed before any data is seen -------------------------
ALPHA = 0.05
POWER = 0.80
N_PER_ARM = 80_000             # sized from the power analysis below, not guessed
DAYS = 14
BASE_CONVERSION = 0.082         # historical checkout conversion
TRUE_LIFT_REL = 0.06            # ground truth: a 6% relative lift (unknown to the analyst)
PRE_REGISTERED_MIN_LIFT = 0.02  # we only ship if we are convinced the lift beats +2% relative


# ============================================================== 1. design
def required_n(p: float, rel_lift: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Per-arm sample size for a two-sided two-proportion test."""
    p2 = p * (1 + rel_lift)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p + p2) / 2
    return int(np.ceil((z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b *
                        np.sqrt(p * (1 - p) + p2 * (1 - p2))) ** 2 / (p2 - p) ** 2))


def mde(p: float, n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest relative lift detectable with n per arm."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    abs_mde = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n)
    return float(abs_mde / p)


# ============================================================== simulation
def simulate_experiment() -> pd.DataFrame:
    """User-level data with a pre-period covariate, so CUPED has something to use."""
    n = N_PER_ARM * 2
    arm = np.where(np.arange(n) % 2 == 0, "control", "treatment")
    day = rng.integers(1, DAYS + 1, n)

    # latent user quality drives both the pre-period and the in-experiment behaviour
    quality = rng.normal(0, 1, n)
    pre_sessions = rng.poisson(np.exp(0.6 + 0.35 * quality))
    # spend in the 4 weeks before the test: driven by the same latent quality, which is
    # what makes it a useful CUPED covariate
    pre_revenue = pre_sessions * rng.gamma(6.0, 4.0, n) * np.exp(0.9 * quality)

    p = BASE_CONVERSION * np.exp(0.45 * quality)
    p = np.clip(p * np.where(arm == "treatment", 1 + TRUE_LIFT_REL, 1.0), 0.001, 0.95)
    converted = rng.random(n) < p

    sessions = 1 + rng.poisson(np.exp(0.5 + 0.3 * quality))
    order_value = rng.gamma(6.0, 9.0, n) * np.exp(0.8 * quality)
    revenue = np.where(converted, order_value, 0.0)

    return pd.DataFrame({
        "user_id": np.arange(n),
        "arm": arm,
        "day": day,
        "pre_sessions": pre_sessions,
        "pre_revenue": pre_revenue.round(2),
        "sessions": sessions,
        "converted": converted.astype(int),
        "revenue": revenue.round(2),
    })


# ============================================================== 3. inference
def two_proportion_test(x1, n1, x2, n2) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pooled
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    half = stats.norm.ppf(1 - ALPHA / 2) * se_unpooled
    return {
        "control_rate": float(p1), "treatment_rate": float(p2),
        "abs_diff": float(p2 - p1), "rel_lift": float((p2 - p1) / p1),
        "z": float(z), "p_value": float(pval),
        "ci95_abs": [float(p2 - p1 - half), float(p2 - p1 + half)],
        "ci95_rel": [float((p2 - p1 - half) / p1), float((p2 - p1 + half) / p1)],
    }


def bootstrap_diff(a, b, n_boot=4000) -> dict:
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (rng.choice(b, len(b), replace=True).mean()
                   - rng.choice(a, len(a), replace=True).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
             "p_value_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}


def delta_method_ratio(num_a, den_a, num_b, den_b) -> dict:
    """Variance of a ratio metric (revenue per session) when the denominator is random."""
    def ratio_var(num, den):
        n = len(num)
        r = num.sum() / den.sum()
        mu_d = den.mean()
        var = (num.var(ddof=1) - 2 * r * np.cov(num, den, ddof=1)[0, 1]
               + r ** 2 * den.var(ddof=1)) / (n * mu_d ** 2)
        return r, var
    r_a, v_a = ratio_var(num_a, den_a)
    r_b, v_b = ratio_var(num_b, den_b)
    se = np.sqrt(v_a + v_b)
    diff = r_b - r_a
    z = diff / se
    half = stats.norm.ppf(1 - ALPHA / 2) * se
    return {"control": float(r_a), "treatment": float(r_b), "diff": float(diff),
            "rel_lift": float(diff / r_a), "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
            "ci95": [float(diff - half), float(diff + half)]}


# ============================================================== 4. CUPED
def cuped(y: np.ndarray, x: np.ndarray, arm: np.ndarray) -> dict:
    """Y adjusted by a pre-experiment covariate: same expectation, smaller variance."""
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    a, b = arm == "control", arm == "treatment"
    raw = stats.ttest_ind(y[b], y[a], equal_var=False)
    adj = stats.ttest_ind(y_adj[b], y_adj[a], equal_var=False)

    def ci(v_b, v_a):
        d = v_b.mean() - v_a.mean()
        se = np.sqrt(v_b.var(ddof=1) / len(v_b) + v_a.var(ddof=1) / len(v_a))
        h = stats.norm.ppf(1 - ALPHA / 2) * se
        return d, [float(d - h), float(d + h)], float(se)

    d_raw, ci_raw, se_raw = ci(y[b], y[a])
    d_adj, ci_adj, se_adj = ci(y_adj[b], y_adj[a])
    return {
        "theta": float(theta),
        "correlation_pre_post": float(np.corrcoef(y, x)[0, 1]),
        "variance_reduction_pct": float((1 - y_adj.var(ddof=1) / y.var(ddof=1)) * 100),
        "se_reduction_pct": float((1 - se_adj / se_raw) * 100),
        "raw": {"diff": float(d_raw), "ci95": ci_raw, "p_value": float(raw.pvalue)},
        "cuped": {"diff": float(d_adj), "ci95": ci_adj, "p_value": float(adj.pvalue)},
        "effective_sample_size_multiplier": float((se_raw / se_adj) ** 2),
    }


# ============================================================== 5. peeking
def peeking_study(n_sims=3000, looks=7, p=BASE_CONVERSION, n_per_arm=15_000) -> dict:
    """How often a null experiment (A/A) crosses p<0.05 if we check every other day."""
    steps = np.linspace(n_per_arm / looks, n_per_arm, looks).astype(int)
    crossed_fixed = 0
    crossed_any = 0
    max_abs_z = np.empty(n_sims)
    for s in range(n_sims):
        a = rng.random(n_per_arm) < p
        b = rng.random(n_per_arm) < p
        zs = []
        for k in steps:
            x1, x2 = a[:k].sum(), b[:k].sum()
            pooled = (x1 + x2) / (2 * k)
            se = np.sqrt(pooled * (1 - pooled) * 2 / k)
            zs.append(0.0 if se == 0 else (x2 / k - x1 / k) / se)
        zs = np.abs(np.array(zs))
        max_abs_z[s] = zs.max()
        crossed_any += zs.max() > stats.norm.ppf(1 - ALPHA / 2)
        crossed_fixed += zs[-1] > stats.norm.ppf(1 - ALPHA / 2)
    # a Pocock-style constant boundary calibrated so that ANY look keeps family-wise alpha
    z_boundary = float(np.quantile(max_abs_z, 1 - ALPHA))
    return {
        "looks": looks,
        "false_positive_rate_fixed_horizon": crossed_fixed / n_sims,
        "false_positive_rate_with_peeking": crossed_any / n_sims,
        "naive_boundary_z": float(stats.norm.ppf(1 - ALPHA / 2)),
        "calibrated_sequential_boundary_z": z_boundary,
        "n_simulations": n_sims,
    }


# ============================================================== 7. Bayesian
def bayesian_decision(x1, n1, x2, n2, n_draws=200_000) -> dict:
    """Beta-Binomial with a uniform prior; decision rule stated in business terms."""
    post_a = rng.beta(1 + x1, 1 + n1 - x1, n_draws)
    post_b = rng.beta(1 + x2, 1 + n2 - x2, n_draws)
    rel = (post_b - post_a) / post_a
    return {
        "p_treatment_better": float((post_b > post_a).mean()),
        "p_rel_lift_above_pre_registered": float((rel > PRE_REGISTERED_MIN_LIFT).mean()),
        "rel_lift_posterior_mean": float(rel.mean()),
        "rel_lift_cred_interval_95": [float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))],
        "expected_loss_if_ship": float(np.maximum(post_a - post_b, 0).mean()),
        "expected_loss_if_not_ship": float(np.maximum(post_b - post_a, 0).mean()),
    }


def benjamini_hochberg(pvals: dict, alpha: float = ALPHA) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, max_sig = {}, -1
    for i, (k, p) in enumerate(items, start=1):
        if p <= i / m * alpha:
            max_sig = i
    for i, (k, p) in enumerate(items, start=1):
        out[k] = {"p_value": p, "bh_threshold": i / m * alpha, "significant_after_bh": i <= max_sig}
    return out


def main() -> None:
    t0 = time.time()
    set_style()
    print("[1/7] design: power and minimum detectable effect")
    n_needed = required_n(BASE_CONVERSION, PRE_REGISTERED_MIN_LIFT)
    design = {
        "baseline_conversion": BASE_CONVERSION,
        "alpha": ALPHA, "power": POWER,
        "n_per_arm_planned": N_PER_ARM,
        "mde_rel_at_planned_n": mde(BASE_CONVERSION, N_PER_ARM),
        "n_per_arm_needed_for_2pct_lift": n_needed,
        "weeks_needed_at_planned_traffic": round(n_needed / (N_PER_ARM / (DAYS / 7)), 1),
    }
    print("  with %d users/arm the MDE is %.1f%% relative; detecting a %.0f%% lift would need "
          "%s users/arm" % (N_PER_ARM, design["mde_rel_at_planned_n"] * 100,
                            PRE_REGISTERED_MIN_LIFT * 100, format(n_needed, ",")))

    df = simulate_experiment()
    a = df[df.arm == "control"]
    b = df[df.arm == "treatment"]
    print("[2/7] sanity: sample ratio mismatch")
    counts = df.arm.value_counts()
    srm_chi2, srm_p = stats.chisquare([counts["control"], counts["treatment"]])
    print("  %d control vs %d treatment, SRM chi2 p = %.3f -> %s"
          % (counts["control"], counts["treatment"], srm_p,
             "assignment looks healthy" if srm_p > 0.01 else "STOP: assignment is broken"))

    print("[3/7] inference on the primary and secondary metrics")
    primary = two_proportion_test(a.converted.sum(), len(a), b.converted.sum(), len(b))
    print("  conversion %.3f%% -> %.3f%%  (%+.1f%% relative, p = %.4f, 95%% CI [%+.1f%%, %+.1f%%])"
          % (primary["control_rate"] * 100, primary["treatment_rate"] * 100,
             primary["rel_lift"] * 100, primary["p_value"],
             primary["ci95_rel"][0] * 100, primary["ci95_rel"][1] * 100))
    arpu = bootstrap_diff(a.revenue.values, b.revenue.values)
    print("  revenue per user %+.2f USD, bootstrap 95%% CI [%+.2f, %+.2f]"
          % (arpu["diff"], arpu["ci95"][0], arpu["ci95"][1]))
    rps = delta_method_ratio(a.revenue.values, a.sessions.values,
                             b.revenue.values, b.sessions.values)
    print("  revenue per session %+.3f USD (delta method, p = %.4f)" % (rps["diff"], rps["p_value"]))

    print("[4/7] CUPED variance reduction on revenue per user")
    cup = cuped(df.revenue.values, df.pre_revenue.values, df.arm.values)
    print("  pre/post correlation %.3f -> variance -%.1f%%, standard error -%.1f%% "
          "(equivalent to %.2fx the traffic)"
          % (cup["correlation_pre_post"], cup["variance_reduction_pct"],
             cup["se_reduction_pct"], cup["effective_sample_size_multiplier"]))
    print("  p-value %.4f -> %.4f after adjustment" % (cup["raw"]["p_value"], cup["cuped"]["p_value"]))

    print("[5/7] peeking: what daily checks do to the false-positive rate")
    peek = peeking_study()
    print("  A/A false positives: %.1f%% at a fixed horizon vs %.1f%% when peeking %d times"
          % (peek["false_positive_rate_fixed_horizon"] * 100,
             peek["false_positive_rate_with_peeking"] * 100, peek["looks"]))
    print("  boundary needed to keep alpha at 5%% across all looks: |z| > %.2f (naive uses 1.96)"
          % peek["calibrated_sequential_boundary_z"])

    print("[6/7] multiple metrics: Benjamini-Hochberg")
    guardrails = {
        "conversion": primary["p_value"],
        "revenue_per_user": arpu["p_value_two_sided"],
        "revenue_per_session": rps["p_value"],
        "sessions_per_user": float(stats.ttest_ind(b.sessions, a.sessions, equal_var=False).pvalue),
        "pre_period_revenue_balance": float(
            stats.ttest_ind(b.pre_revenue, a.pre_revenue, equal_var=False).pvalue),
    }
    bh = benjamini_hochberg(guardrails)
    for k, v in sorted(bh.items(), key=lambda kv: kv[1]["p_value"]):
        print("  %-28s p = %.4f  BH threshold %.4f  %s"
              % (k, v["p_value"], v["bh_threshold"],
                 "significant" if v["significant_after_bh"] else "not significant"))

    print("[7/7] decision")
    bayes = bayesian_decision(a.converted.sum(), len(a), b.converted.sum(), len(b))
    ship = (bayes["p_rel_lift_above_pre_registered"] > 0.90
            and primary["p_value"] < ALPHA
            and srm_p > 0.01)
    print("  P(treatment > control) = %.3f | P(lift > %.0f%%) = %.3f | expected loss if we ship "
          "= %.5f pp" % (bayes["p_treatment_better"], PRE_REGISTERED_MIN_LIFT * 100,
                         bayes["p_rel_lift_above_pre_registered"],
                         bayes["expected_loss_if_ship"] * 100))
    print("  pre-registered rule -> %s" % ("SHIP" if ship else "DO NOT SHIP"))
    print("  (ground truth used by the simulator: %+.0f%% relative lift)" % (TRUE_LIFT_REL * 100))

    # ------------------------------------------------------------- figures
    ns = np.unique(np.linspace(2000, 60000, 40).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    axes[0].plot(ns, [mde(BASE_CONVERSION, n) * 100 for n in ns], color=PALETTE[0])
    axes[0].axvline(N_PER_ARM, ls="--", color=PALETTE[4], label="planned n")
    axes[0].axhline(PRE_REGISTERED_MIN_LIFT * 100, ls=":", color=PALETTE[2],
                    label="target lift %.0f%%" % (PRE_REGISTERED_MIN_LIFT * 100))
    axes[0].set(title="Minimum detectable effect", xlabel="users per arm",
                ylabel="detectable relative lift (%)")
    axes[0].legend(fontsize=8)

    lifts = np.linspace(0.01, 0.12, 40)
    for n in [6000, 12000, 25000]:
        powers = []
        for lift in lifts:
            p2 = BASE_CONVERSION * (1 + lift)
            se = np.sqrt(BASE_CONVERSION * (1 - BASE_CONVERSION) / n + p2 * (1 - p2) / n)
            z = (p2 - BASE_CONVERSION) / se
            powers.append(stats.norm.cdf(z - stats.norm.ppf(1 - ALPHA / 2)))
        axes[1].plot(lifts * 100, np.array(powers) * 100, label="n = %s/arm" % format(n, ","))
    axes[1].axhline(80, ls="--", color=PALETTE[5], lw=1)
    axes[1].set(title="Power curves", xlabel="true relative lift (%)", ylabel="power (%)")
    axes[1].legend(fontsize=8)

    daily = (df.assign(cum=1).sort_values("day")
             .groupby(["day", "arm"]).converted.agg(["sum", "size"]).reset_index())
    piv = daily.pivot(index="day", columns="arm", values=["sum", "size"]).cumsum()
    zs = []
    for d in piv.index:
        x1, n1 = piv.loc[d, ("sum", "control")], piv.loc[d, ("size", "control")]
        x2, n2 = piv.loc[d, ("sum", "treatment")], piv.loc[d, ("size", "treatment")]
        zs.append(two_proportion_test(x1, n1, x2, n2)["z"])
    axes[2].plot(piv.index, zs, marker="o", color=PALETTE[1], label="observed z")
    axes[2].axhline(1.96, ls="--", color=PALETTE[5], label="naive 1.96")
    axes[2].axhline(peek["calibrated_sequential_boundary_z"], ls="--", color=PALETTE[4],
                    label="sequential %.2f" % peek["calibrated_sequential_boundary_z"])
    axes[2].set(title="Daily monitoring of the test statistic", xlabel="day", ylabel="z")
    axes[2].legend(fontsize=8)
    save(fig, FIG / "01_design_and_monitoring.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    axes[0].bar(["raw", "CUPED"], [cup["raw"]["ci95"][1] - cup["raw"]["ci95"][0],
                                   cup["cuped"]["ci95"][1] - cup["cuped"]["ci95"][0]],
                color=[PALETTE[5], PALETTE[0]])
    axes[0].set(title="95% CI width for revenue per user", ylabel="USD")
    for i, v in enumerate([cup["raw"]["ci95"][1] - cup["raw"]["ci95"][0],
                           cup["cuped"]["ci95"][1] - cup["cuped"]["ci95"][0]]):
        axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom")

    axes[1].bar(["fixed\nhorizon", "7 peeks\nnaive", "7 peeks\ncalibrated"],
                [peek["false_positive_rate_fixed_horizon"] * 100,
                 peek["false_positive_rate_with_peeking"] * 100, ALPHA * 100],
                color=[PALETTE[0], PALETTE[4], PALETTE[2]])
    axes[1].axhline(5, ls="--", color="#222", lw=1, label="nominal 5%")
    axes[1].set(title="A/A false-positive rate", ylabel="%")
    axes[1].legend(fontsize=8)

    post_a = rng.beta(1 + a.converted.sum(), 1 + len(a) - a.converted.sum(), 60000)
    post_b = rng.beta(1 + b.converted.sum(), 1 + len(b) - b.converted.sum(), 60000)
    axes[2].hist(post_a * 100, bins=80, alpha=0.6, density=True, label="control")
    axes[2].hist(post_b * 100, bins=80, alpha=0.6, density=True, label="treatment")
    axes[2].set(title="Posterior conversion rate", xlabel="conversion (%)")
    axes[2].legend(fontsize=8)
    save(fig, FIG / "02_cuped_peeking_posterior.png")

    metrics = {
        "simulation_ground_truth": {"true_rel_lift": TRUE_LIFT_REL,
                                    "base_conversion": BASE_CONVERSION,
                                    "n_per_arm": N_PER_ARM, "days": DAYS},
        "design": design,
        "srm_check": {"control": int(counts["control"]), "treatment": int(counts["treatment"]),
                      "chi2": float(srm_chi2), "p_value": float(srm_p)},
        "primary_metric_conversion": primary,
        "revenue_per_user_bootstrap": arpu,
        "revenue_per_session_delta_method": rps,
        "cuped": cup,
        "peeking": peek,
        "multiple_testing_bh": bh,
        "bayesian": bayes,
        "decision": {"ship": bool(ship),
                     "rule": "ship if p < 0.05, P(rel lift > 2%) > 0.90 and no SRM"},
    }
    (PROJ / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.to_csv(PROJ / "data" / "ab_test_users.csv", index=False)
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
