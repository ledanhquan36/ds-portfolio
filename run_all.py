"""Run every project end to end and print a summary table.

    python run_all.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = [
    ("01 churn prediction", "01_churn_prediction/src/train.py"),
    ("02 demand forecasting", "02_demand_forecasting/src/forecast.py"),
    ("03 text classification", "03_review_nlp/src/text_classification.py"),
    ("04 a/b test analysis", "04_ab_testing/src/ab_test_analysis.py"),
]


def main() -> None:
    results = []
    for name, script in PROJECTS:
        print("\n" + "=" * 78 + "\n== %s\n" % name + "=" * 78)
        t = time.time()
        # no cwd= on purpose: every script resolves its own paths from __file__,
        # so this works no matter where it is launched from
        proc = subprocess.run([sys.executable, str(ROOT / script)])
        results.append((name, proc.returncode, time.time() - t))

    print("\n" + "=" * 78)
    for name, code, secs in results:
        print("%-26s %-6s %5.1fs" % (name, "ok" if code == 0 else "FAILED", secs))

    print("\nheadline metrics")
    paths = {
        "churn test ROC-AUC": ("01_churn_prediction", ["test", "roc_auc"]),
        "churn campaign value (USD/1k customers)": ("01_churn_prediction",
                                                    ["business", "per_1000_customers_usd"]),
        "forecast MAE (rides/hour)": ("02_demand_forecasting", ["models_test", "hgb_poisson", "mae"]),
        "forecast gain vs seasonal naive (%)": ("02_demand_forecasting",
                                                ["improvement_vs_seasonal_naive_pct"]),
        "text macro-F1 (clean)": ("03_review_nlp", ["best_model_test", "macro_f1"]),
        "macro-F1 inflation from leakage": ("03_review_nlp", ["macro_f1_inflation_from_metadata"]),
        "A/B detected relative lift": ("04_ab_testing", ["primary_metric_conversion", "rel_lift"]),
        "A/B peeking false-positive rate": ("04_ab_testing",
                                            ["peeking", "false_positive_rate_with_peeking"]),
    }
    for label, (proj, keys) in paths.items():
        f = ROOT / proj / "reports" / "metrics.json"
        if not f.exists():
            continue
        v = json.loads(f.read_text(encoding="utf-8"))
        for k in keys:
            v = v[k]
        print("  %-42s %s" % (label, round(v, 4) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
