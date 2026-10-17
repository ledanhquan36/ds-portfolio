"""Domain feature engineering for the churn model."""
from __future__ import annotations

import numpy as np
import pandas as pd

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDONS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # how many paid add-ons the customer actually holds -> stickiness proxy
    out["n_addons"] = sum((out[c] == "Yes").astype(int) for c in ADDONS)
    # automatic payment is a known retention lever
    out["is_autopay"] = out["PaymentMethod"].str.contains("automatic").astype(int)
    # lifecycle stage matters more than raw months
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)
    # realised ARPU vs current price: positive means the customer was upsold recently
    out["arpu_hist"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["price_delta"] = out["MonthlyCharges"] - out["arpu_hist"]
    # spend intensity relative to the book
    out["charges_per_addon"] = out["MonthlyCharges"] / (1 + out["n_addons"])
    return out


NUMERIC = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "n_addons",
    "is_autopay",
    "arpu_hist",
    "price_delta",
    "charges_per_addon",
]
CATEGORICAL = [*RAW_CATEGORICAL, "tenure_bucket"]


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    return num, cat
