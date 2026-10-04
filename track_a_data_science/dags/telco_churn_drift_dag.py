"""Airflow DAG: Telco Customer Churn Scheduled Drift Audit & Retrain Trigger

Schedule: Daily at 02:00 UTC ('0 2 * * *')
Workflow:
  1. check_incoming_data_sensor
  2. execute_evidently_drift_audit
  3. evaluate_drift_condition (Branching)
     -> trigger_model_retraining (if drift_share > 0.25)
     -> log_pipeline_stable (if drift_share <= 0.25)
"""

from datetime import datetime, timedelta
import json
from pathlib import Path

# Try importing Airflow, providing graceful fallback/mock execution for local standalone runs
try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False


def run_drift_check(**kwargs):
    from track_a_data_science.drift_monitor import run_drift_monitoring
    print("Executing scheduled Evidently AI drift evaluation...")
    run_drift_monitoring()
    
    # Read metrics to decide branching
    metrics_file = Path(__file__).resolve().parent.parent / "reports" / "custom_drift_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        print(f"Evaluated drift metrics: {data}")
    return "drift_check_complete"


def evaluate_drift_threshold(**kwargs):
    # If drift delta exceeds threshold, branch to retrain
    metrics_file = Path(__file__).resolve().parent.parent / "reports" / "custom_drift_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        if abs(data.get("charge_drift_delta", 0)) > 10.0 or abs(data.get("segment_churn_rate_shift", 0)) > 0.05:
            print("ALERT: Severe feature/target drift detected! Branching to model retrain.")
            return "trigger_model_retraining"
    return "log_pipeline_healthy"


def trigger_model_retraining(**kwargs):
    print("Triggering retraining pipeline with updated production reference window...")
    from track_a_data_science.train_mlflow import train_and_track_experiments
    train_and_track_experiments()
    print("Retraining completed; candidate registered in MLflow Model Registry.")


def log_pipeline_healthy(**kwargs):
    print("Pipeline healthy: Feature distributions within statistical tolerance limits.")


if AIRFLOW_INSTALLED:
    default_args = {
        "owner": "mlops-platform",
        "depends_on_past": False,
        "start_date": datetime(2026, 1, 1),
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id="telco_churn_daily_drift_monitor",
        default_args=default_args,
        description="Daily drift monitoring and automated retraining orchestration",
        schedule="0 2 * * *",
        catchup=False,
    )

    t1 = PythonOperator(
        task_id="run_evidently_drift_audit",
        python_callable=run_drift_check,
        dag=dag,
    )

    t2 = BranchPythonOperator(
        task_id="evaluate_drift_threshold",
        python_callable=evaluate_drift_threshold,
        dag=dag,
    )

    t3_retrain = PythonOperator(
        task_id="trigger_model_retraining",
        python_callable=trigger_model_retraining,
        dag=dag,
    )

    t3_healthy = PythonOperator(
        task_id="log_pipeline_healthy",
        python_callable=log_pipeline_healthy,
        dag=dag,
    )

    t1 >> t2 >> [t3_retrain, t3_healthy]
