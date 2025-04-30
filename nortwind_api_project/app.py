from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# FastAPI başlat
app = FastAPI()

# Model ve scaler'ı yükle
model = load_model("../models/reorder_model.h5")
scaler = joblib.load("../models/reorder_scaler.pkl")

# API'ye gönderilecek verinin şeması
class CustomerFeatures(BaseModel):
    total_spent: float
    total_orders: int
    avg_order_value: float
    days_since_last_order: int
    last_order_month: int
    last_order_season: int

# Tahmin endpoint'i
@app.post("/reorder_predict")
def predict(customer: CustomerFeatures):
    data = pd.DataFrame([customer.dict()])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)[0][0]
    return {
        "reorder_probability": float(prediction),
        "will_order": int(prediction > 0.5)
    }


"""
Example Re-Order:
{
  "total_spent": 500.0,
  "total_orders": 10,
  "avg_order_value": 50.0,
  "days_since_last_order": 30,
  "last_order_month": 3,
  "last_order_season": 1
}


"""
