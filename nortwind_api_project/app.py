from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

# Initialize FastAPI application
app = FastAPI()

# Define the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the path to the trained Keras model
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "reorder_model.h5")

# Set the path to the saved scaler (used to normalize input data)
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "reorder_scaler.pkl")

# Load the trained deep learning model
model = load_model(MODEL_PATH)

# Load the scaler used during training to transform the input data
scaler = joblib.load(SCALER_PATH)

# Define the expected input data schema using Pydantic
class CustomerFeatures(BaseModel):
    total_spent: float
    total_orders: int
    avg_order_value: float
    days_since_last_order: int
    last_order_month: int
    last_order_season: int

# Define the API endpoint to make reorder predictions
@app.post("/reorder_predict")
def predict(customer: CustomerFeatures):
    # Convert the input data to a DataFrame
    data = pd.DataFrame([customer.dict()])

    # Normalize the data using the loaded scaler
    scaled = scaler.transform(data)

    # Predict using the loaded deep learning model
    prediction = model.predict(scaled)[0][0]
    return {
        "will_order": int(prediction > 0.5)
    }

"""
Example Re-Order Request Body:
{
  "total_spent": 500.0,
  "total_orders": 10,
  "avg_order_value": 50.0,
  "days_since_last_order": 30,
  "last_order_month": 3,
  "last_order_season": 1
}
"""
