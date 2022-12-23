import os
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        raise RuntimeError
    model = joblib.load(model_path)


@app.get("/")
def root():
    """Check if program is up running"""
    return {"process": "working"}


@app.get("/health")
def health() -> bool:
    """Check if model is loaded"""
    return not (model is None)


@app.get("/predict")
async def predict():
    """Prediction happens here"""
    x_val_path = os.getenv("PATH_TO_DATA")
    x_val = pd.read_csv(x_val_path)
    status = "not ready"
    y_pred = model.predict(x_val)
    if y_pred[0]:
        status = "done"
    return {"prediction": status}
