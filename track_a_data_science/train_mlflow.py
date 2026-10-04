import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = Path(__file__).resolve().parent / "mlflow.db"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor


def train_and_track_experiments():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # SQLite backend for robust tracking and full model registry support
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")
    mlflow.set_experiment("Telco_Customer_Churn_Prediction")

    ref_path = DATA_DIR / "reference_data.csv"
    if not ref_path.exists():
        from prepare_data import load_and_clean_data, split_and_inject_drift
        df = load_and_clean_data()
        split_and_inject_drift(df)

    data = pd.read_csv(ref_path)
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    # Model configurations to experiment on
    models_config = [
        {
            "name": "Logistic_Regression_L2",
            "model": LogisticRegression(C=0.1, penalty="l2", max_iter=500, random_state=42),
            "params": {"C": 0.1, "penalty": "l2", "model_family": "Linear"},
        },
        {
            "name": "Random_Forest_Balanced",
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
            ),
            "params": {"n_estimators": 100, "max_depth": 6, "class_weight": "balanced", "model_family": "TreeEnsemble"},
        },
        {
            "name": "Gradient_Boosting_Tuned",
            "model": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
            ),
            "params": {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 4, "model_family": "Boosting"},
        },
    ]

    runs_summary = []

    for cfg in models_config:
        model_name = cfg["name"]
        print(f"\n--- Training {model_name} ---")

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", cfg["model"]),
            ])

            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            # Compute metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_prob)

            print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")

            # Log to MLflow
            mlflow.log_params(cfg["params"])
            mlflow.log_metrics({
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
            })

            # Plot & log Confusion Matrix
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_estimator(pipeline, X_val, y_val, ax=ax, cmap="Blues")
            cm_path = ARTIFACTS_DIR / f"{model_name}_cm.png"
            plt.title(f"Confusion Matrix: {model_name}")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # Plot & log ROC Curve
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_estimator(pipeline, X_val, y_val, ax=ax)
            roc_path = ARTIFACTS_DIR / f"{model_name}_roc.png"
            plt.title(f"ROC Curve: {model_name} (AUC={auc:.3f})")
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            # Log model with cloudpickle serialization for complete tree ensemble support
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs_summary.append({
                "run_id": run_id,
                "model_name": model_name,
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc,
                "recall": rec,
                "precision": prec,
            })

    # Summary table
    df_runs = pd.DataFrame(runs_summary).sort_values(by="roc_auc", ascending=False)
    print("\n=== Experiment Run Comparison ===")
    print(df_runs.to_string(index=False))

    # Register best-performing model by ROC-AUC & F1
    best_run = df_runs.iloc[0]
    print(f"\nWinning Model: {best_run['model_name']} (ROC-AUC: {best_run['roc_auc']:.4f}, F1: {best_run['f1_score']:.4f})")

    model_uri = f"runs:/{best_run['run_id']}/model"
    registered_model_name = "TelcoChurnProductionModel"
    reg_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    client = MlflowClient()
    # Transition through stages: Staging -> Production
    print(f"Transitioning model version {reg_model.version} to 'Staging'...")
    client.transition_model_version_stage(
        name=registered_model_name,
        version=reg_model.version,
        stage="Staging",
    )

    print(f"Promoting model version {reg_model.version} to 'Production'...")
    client.transition_model_version_stage(
        name=registered_model_name,
        version=reg_model.version,
        stage="Production",
    )

    # Save summary report table
    summary_file = ARTIFACTS_DIR / "run_comparison_table.csv"
    df_runs.to_csv(summary_file, index=False)
    print(f"Saved run comparison table to {summary_file}")
    return df_runs


if __name__ == "__main__":
    train_and_track_experiments()
