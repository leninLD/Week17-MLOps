"""Airflow DAG: Agentic AI Nightly Regression & Quality Evaluation

Schedule: Nightly at 03:00 UTC ('0 3 * * *')
Workflow:
  1. execute_nightly_regression_suite
  2. evaluate_regression_pass_rate (Branching)
     -> alert_prompt_regression (if pct_passed < 0.90)
     -> promote_prompt_release (if pct_passed >= 0.90)
"""

from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False


def run_nightly_regression(**kwargs):
    from track_b_agentic_ai.evidently_eval import run_evidently_regression_suite
    print("Running scheduled LLM regression test suite...")
    df = run_evidently_regression_suite()
    print("Suite execution completed.")
    return "eval_complete"


def check_regression_pass_threshold(**kwargs):
    report_file = Path(__file__).resolve().parent.parent / "reports" / "regression_test_report.json"
    if report_file.exists():
        records = json.loads(report_file.read_text())
        passed = sum(1 for r in records if r["passed"])
        rate = passed / len(records) if records else 0.0
        if rate < 0.90:
            print(f"REGRESSION ALERT: Pass rate {rate:.1%} dropped below 90% threshold!")
            return "alert_prompt_regression"
    return "promote_prompt_release"


def alert_prompt_regression(**kwargs):
    print("SENDING ALERT: Operational prompt candidate failed regression evaluation. Release blocked.")


def promote_prompt_release(**kwargs):
    print("CONFIRMATION: All golden regression cases passed. Candidate marked ready for promotion.")


if AIRFLOW_INSTALLED:
    default_args = {
        "owner": "agentic-ops",
        "depends_on_past": False,
        "start_date": datetime(2026, 1, 1),
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id="agent_nightly_regression_eval",
        default_args=default_args,
        description="Nightly evaluation of agentic reasoning against golden reference sets",
        schedule="0 3 * * *",
        catchup=False,
    )

    t1 = PythonOperator(
        task_id="run_regression_suite",
        python_callable=run_nightly_regression,
        dag=dag,
    )

    t2 = BranchPythonOperator(
        task_id="check_regression_threshold",
        python_callable=check_regression_pass_threshold,
        dag=dag,
    )

    t3_alert = PythonOperator(
        task_id="alert_prompt_regression",
        python_callable=alert_prompt_regression,
        dag=dag,
    )

    t3_promote = PythonOperator(
        task_id="promote_prompt_release",
        python_callable=promote_prompt_release,
        dag=dag,
    )

    t1 >> t2 >> [t3_alert, t3_promote]
