from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "telco_customer_churn.csv"


def load_and_clean_data(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean numeric fields
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customerID (non-predictive identifier)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Binary target encoding: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def split_and_inject_drift(
    df: pd.DataFrame,
    ref_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into 70% reference (training-time baseline) and 30% current 
    (incoming production data with intentionally injected synthetic drift).
    """
    np.random.seed(seed)
    n = len(df)
    shuffled_idx = np.random.permutation(n)
    split_point = int(n * ref_ratio)

    ref_idx = shuffled_idx[:split_point]
    cur_idx = shuffled_idx[split_point:]

    ref_df = df.iloc[ref_idx].copy().reset_index(drop=True)
    cur_df = df.iloc[cur_idx].copy().reset_index(drop=True)

    # --- SYNTHETIC DRIFT INJECTION ---
    # 1. Numeric Feature Drift: Shift 'MonthlyCharges' upwards by $18.50 with random Gaussian noise
    noise = np.random.normal(loc=18.5, scale=4.0, size=len(cur_df))
    cur_df["MonthlyCharges"] = (cur_df["MonthlyCharges"] + noise).clip(lower=15.0)

    # 2. Numeric Feature Drift: Shift 'tenure' towards newer, shorter customers
    cur_df["tenure"] = (cur_df["tenure"] * 0.65).astype(int)

    # 3. Categorical Feature Drift: Skew 'Contract' distribution heavily towards 'Month-to-month'
    skew_mask = np.random.rand(len(cur_df)) < 0.40
    cur_df.loc[skew_mask, "Contract"] = "Month-to-month"

    # 4. Target Label Drift: Slightly increase churn rate by flipping 15% of non-churners
    flip_mask = (cur_df["Churn"] == 0) & (np.random.rand(len(cur_df)) < 0.15)
    cur_df.loc[flip_mask, "Churn"] = 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(DATA_DIR / "reference_data.csv", index=False)
    cur_df.to_csv(DATA_DIR / "current_production_data.csv", index=False)

    print(f"Data prepared successfully:")
    print(f"  Reference set (baseline): {len(ref_df)} rows (Churn rate: {ref_df['Churn'].mean():.2%})")
    print(f"  Current set (with drift): {len(cur_df)} rows (Churn rate: {cur_df['Churn'].mean():.2%})")
    return ref_df, cur_df


if __name__ == "__main__":
    df = load_and_clean_data()
    split_and_inject_drift(df)
