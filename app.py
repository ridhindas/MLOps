import os
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Use a cross-platform relative path
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_URI = os.path.join(base_dir, "mlruns", "1", "models", "m-5af3f43b4d91402e832e97232a3c71e3", "artifacts")

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception:
    model = None

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")
        
    # FIX: Use model_dump() for Pydantic V2 compatibility
    data_df = pd.DataFrame([features.model_dump().values()], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    prediction = model.predict(data_df)
    
    # FIX: Return "prediction" key to match test assertion
    return {"prediction": int(prediction[0])}
