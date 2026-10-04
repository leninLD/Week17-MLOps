from pathlib import Path
from typing import Dict, List, Union
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

DB_PATH = Path(__file__).resolve().parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")

app = FastAPI(
    title="Telco Churn Inference Service",
    version="1.0.0",
    description="Production model serving API loading registered churn models from MLflow Model Registry",
)

MODEL_NAME = "TelcoChurnProductionModel"
STAGE = "Production"
model = None


@app.on_event("startup")
def load_production_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded {model_uri}")
    except Exception as e:
        print(f"Warning: Could not load from registry stage {STAGE}: {e}")
        # Fallback to latest run artifact if registry stage transition is pending
        runs_dir = Path(__file__).resolve().parent / "artifacts" / "run_comparison_table.csv"
        if runs_dir.exists():
            df_runs = pd.read_csv(runs_dir)
            best_run_id = df_runs.iloc[0]["run_id"]
            fallback_uri = f"runs:/{best_run_id}/model"
            model = mlflow.sklearn.load_model(fallback_uri)
            print(f"Loaded fallback model from {fallback_uri}")


class CustomerPayload(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 78.5
    TotalCharges: float = 942.0


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_tier: str
    model_version_stage: str


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "served_model": MODEL_NAME,
        "stage": STAGE,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Inference model is not currently loaded.")

    df_input = pd.DataFrame([customer.model_dump()])

    pred = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0, 1])

    if prob > 0.70:
        risk_tier = "High"
    elif prob > 0.40:
        risk_tier = "Moderate"
    else:
        risk_tier = "Low"

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_tier=risk_tier,
        model_version_stage=STAGE,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8082, reload=False)
