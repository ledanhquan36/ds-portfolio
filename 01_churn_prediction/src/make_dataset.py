"""
Load the Telco customer-churn dataset.

Strategy: try the public IBM CSV first (same schema as the Kaggle "Telco Customer
Churn" set). If there is no network, fall back to a simulator that reproduces the
schema and the documented churn drivers, so the pipeline is reproducible offline.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "telco_churn.csv"
N_CUSTOMERS = 7043
SEED = 42


def _try_download(timeout: int = 15) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(RAW_URL, timeout=timeout) as resp:
            return pd.read_csv(io.BytesIO(resp.read()))
    except Exception as exc:  # offline, proxy, 404...
        print(f"  [data] download failed ({type(exc).__name__}), using simulator")
        return None


def _pick(rng, values, probs, n):
    return rng.choice(values, size=n, p=probs)


def simulate(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """Simulate a telco base whose churn mechanism mirrors the published findings:
    month-to-month contracts, fiber optic, electronic-check payment and short
    tenure drive churn up; tech support / online security and long tenure drive it down.
    """
    rng = np.random.default_rng(seed)

    contract = _pick(rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], n)
    # tenure is contract dependent: long contracts imply customers who already stayed
    base_tenure = rng.gamma(shape=1.5, scale=12, size=n)
    tenure = np.clip(
        base_tenure + np.where(contract == "One year", 14, 0) + np.where(contract == "Two year", 30, 0),
        1,
        72,
    ).round().astype(int)

    internet = _pick(rng, ["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n)
    has_net = internet != "No"

    def addon(p_yes_if_net: float) -> np.ndarray:
        out = np.where(
            has_net,
            rng.choice(["Yes", "No"], size=n, p=[p_yes_if_net, 1 - p_yes_if_net]),
            "No internet service",
        )
        return out

    phone = _pick(rng, ["Yes", "No"], [0.90, 0.10], n)
    multiple = np.where(
        phone == "Yes",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
        "No phone service",
    )

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-CUST" for i in range(n)],
            "gender": _pick(rng, ["Male", "Female"], [0.5, 0.5], n),
            "SeniorCitizen": rng.binomial(1, 0.16, n),
            "Partner": _pick(rng, ["Yes", "No"], [0.48, 0.52], n),
            "Dependents": _pick(rng, ["Yes", "No"], [0.30, 0.70], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": addon(0.40),
            "OnlineBackup": addon(0.44),
            "DeviceProtection": addon(0.44),
            "TechSupport": addon(0.40),
            "StreamingTV": addon(0.50),
            "StreamingMovies": addon(0.50),
            "Contract": contract,
            "PaperlessBilling": _pick(rng, ["Yes", "No"], [0.59, 0.41], n),
            "PaymentMethod": _pick(
                rng,
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                [0.34, 0.23, 0.22, 0.21],
                n,
            ),
        }
    )

    # --- monthly charges follow the product mix -----------------------------
    charges = 19.0
    charges = charges + np.where(df.InternetService == "DSL", 25, 0) + np.where(
        df.InternetService == "Fiber optic", 50, 0
    )
    for col, price in [
        ("OnlineSecurity", 5),
        ("OnlineBackup", 5),
        ("DeviceProtection", 5),
        ("TechSupport", 5),
        ("StreamingTV", 10),
        ("StreamingMovies", 10),
        ("MultipleLines", 6),
    ]:
        charges = charges + np.where(df[col] == "Yes", price, 0)
    df["MonthlyCharges"] = (charges + rng.normal(0, 2.5, n)).clip(18.25, 120).round(2)
    df["TotalCharges"] = (df.MonthlyCharges * df.tenure * rng.normal(1.0, 0.03, n)).round(2)

    # --- churn mechanism ----------------------------------------------------
    z = (
        -1.15
        + 1.35 * (df.Contract == "Month-to-month")
        - 0.85 * (df.Contract == "Two year")
        - 0.030 * df.tenure
        + 0.80 * (df.InternetService == "Fiber optic")
        - 0.45 * (df.InternetService == "No")
        - 0.50 * (df.TechSupport == "Yes")
        - 0.40 * (df.OnlineSecurity == "Yes")
        + 0.45 * (df.PaymentMethod == "Electronic check")
        + 0.012 * (df.MonthlyCharges - df.MonthlyCharges.mean())
        + 0.25 * df.SeniorCitizen
        - 0.15 * (df.Dependents == "Yes")
        + rng.normal(0, 0.45, n)  # irreducible noise -> AUC stays realistic
    )
    p = 1 / (1 + np.exp(-z))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")

    # mirror the real file's quirk: 11 blank TotalCharges for brand-new customers
    blanks = df.index[df.tenure <= 1][:11]
    df.loc[blanks, "TotalCharges"] = " "
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    return df


def load(force_simulate: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"  [data] cached {CSV_PATH.name}")
        return pd.read_csv(CSV_PATH)
    df = None if force_simulate else _try_download()
    source = "public IBM CSV"
    if df is None:
        df = simulate()
        source = "simulated (offline fallback)"
    df.to_csv(CSV_PATH, index=False)
    print(f"  [data] {len(df):,} rows from {source} -> {CSV_PATH.name}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes that both the real and simulated file need."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # blank TotalCharges == brand-new customer, has not been billed yet
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df.drop(columns=["Churn"])


if __name__ == "__main__":
    print(clean(load()).head())
