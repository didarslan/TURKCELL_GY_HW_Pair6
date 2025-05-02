from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
from typing import List

app = FastAPI()

model = tf.keras.models.load_model("customer_model.h5")
scaler = joblib.load("scaler.pkl")

class PredictionRequest(BaseModel):
    categories: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        X = np.array([request.categories])
        X_scaled = scaler.transform(X)
        prob = model.predict(X_scaled)[0][0]
        return {"purchase_probability": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
