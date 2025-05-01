# Nortwind API Project

## Overview

This project is a FastAPI application designed to predict customer reorder behavior using a trained deep learning model. The application exposes an API endpoint that accepts customer features and returns a prediction on whether the customer will place a reorder.

## Features

- **Reorder Prediction**: The API provides a prediction on whether a customer will reorder based on their historical data.
- **Model Integration**: Utilizes a pre-trained Keras model for making predictions.
- **Data Normalization**: Input data is normalized using a pre-saved scaler to ensure consistency with the training data.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd nortwind_api_project
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   uvicorn app:app --reload
   ```

2. **API Endpoint**:
   - **POST** `/reorder_predict`: Accepts customer features and returns a prediction.
   - **Request Body Example**:
     ```json
     {
       "total_spent": 500.0,
       "total_orders": 10,
       "avg_order_value": 50.0,
       "days_since_last_order": 30,
       "last_order_month": 3,
       "last_order_season": 1
     }
     ```

3. **Response**:
   - Returns a JSON object indicating whether the customer will reorder:
     ```json
     {
       "will_order": 0
     }
     ```

## Dependencies

- `pandas==2.2.3`
- `numpy==1.26.4`
- `sqlalchemy==2.0.40`
- `scikit-learn==1.6.1`
- `tensorflow==2.16.2`
- `imbalanced-learn==0.13.0`
- `joblib==1.4.2`
- `fastapi==0.115.12`
- `pydantic==2.11.4`
- `uvicorn==0.34.2` 