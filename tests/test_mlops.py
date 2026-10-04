import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from track_a_data_science.prepare_data import load_and_clean_data, split_and_inject_drift
from track_a_data_science.drift_monitor import calculate_custom_segment_drift
from track_b_agentic_ai.evidently_eval import llm_judge_evaluate, GOLDEN_TEST_SET


def test_data_cleaning():
    csv_path = Path(__file__).resolve().parent.parent / "track_a_data_science" / "data" / "telco_customer_churn.csv"
    if not csv_path.exists():
        pytest.skip("Telco CSV not yet downloaded")
    df = load_and_clean_data(csv_path)
    assert "customerID" not in df.columns
    assert df["Churn"].dtype in [np.int64, int]
    assert df["TotalCharges"].isnull().sum() == 0


def test_drift_injection():
    csv_path = Path(__file__).resolve().parent.parent / "track_a_data_science" / "data" / "telco_customer_churn.csv"
    if not csv_path.exists():
        pytest.skip("Telco CSV not yet downloaded")
    df = load_and_clean_data(csv_path)
    ref, cur = split_and_inject_drift(df, ref_ratio=0.7)

    assert len(ref) > len(cur)
    # Drift verification: MonthlyCharges in current should have higher mean
    assert cur["MonthlyCharges"].mean() > ref["MonthlyCharges"].mean()


def test_custom_segment_drift_metric():
    ref_df = pd.DataFrame({
        "Contract": ["Month-to-month", "Month-to-month", "One year"],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Churn": [0, 1, 0]
    })
    cur_df = pd.DataFrame({
        "Contract": ["Month-to-month", "Month-to-month", "One year"],
        "MonthlyCharges": [75.0, 85.0, 70.0],
        "Churn": [1, 1, 0]
    })
    res = calculate_custom_segment_drift(ref_df, cur_df)
    assert res["metric_name"] == "MonthToMonth_Segment_Anomaly"
    assert res["charge_drift_delta"] == 25.0
    assert res["segment_churn_rate_shift"] == 0.5


def test_llm_judge_evaluator():
    ref = "Tier-1 services require a 99.99% monthly availability target, with a maximum allowable downtime of 4.38 minutes per month."
    good_cand = "Tier-1 requires 99.99% availability with at most 4.38 minutes downtime per month."
    bad_cand = "All servers are completely fine and don't need any maintenance."

    good_res = llm_judge_evaluate(good_cand, ref)
    assert good_res["passed"] is True
    assert good_res["verdict"] == "correct"

    bad_res = llm_judge_evaluate(bad_cand, ref)
    assert bad_res["passed"] is False
    assert bad_res["verdict"] == "incorrect"
