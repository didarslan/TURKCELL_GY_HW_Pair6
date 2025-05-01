import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import random
import os
import joblib

# Connect to the PostgreSQL database
db_url = "postgresql://postgres:***@localhost:***/Gyk1"
engine = create_engine(db_url)

# SQL query: customer-level purchase summary
summary_query = '''
SELECT 
  c.customer_id AS customer_id,
  COUNT(DISTINCT o.order_id) AS total_orders,
  SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_spent,
  AVG(od.unit_price * od.quantity * (1 - od.discount)) AS avg_order_value,
  MAX(o.order_date) AS last_order_date
FROM Customers c
JOIN Orders o ON c.customer_id = o.customer_id
JOIN Order_Details od ON o.order_id = od.order_id
GROUP BY c.customer_id;
'''

# SQL query: raw order data (for label generation)
orders_query = '''
SELECT customer_id, order_date FROM Orders;
'''

# Read data from SQL database
summary_df = pd.read_sql(summary_query, engine)
orders_df = pd.read_sql(orders_query, engine)

# Convert date columns to datetime
summary_df['last_order_date'] = pd.to_datetime(summary_df['last_order_date'])
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

def has_order_in_next_6_months(customer_id, last_date):
    """
    Check if a customer placed a new order within 6 months after their last order.

    Args:
        customer_id (str): ID of the customer
        last_date (datetime): The last order date of the customer

    Returns:
        int: 1 if a new order exists within 6 months, otherwise 0
    """
    future_orders = orders_df[
        (orders_df["customer_id"] == customer_id) &
        (orders_df["order_date"] > last_date) &
        (orders_df["order_date"] <= last_date + pd.Timedelta(days=180))
    ]
    return 1 if not future_orders.empty else 0

# Create target label: whether customer reordered within 6 months
summary_df["label"] = summary_df.apply(
    lambda row: has_order_in_next_6_months(row["customer_id"], row["last_order_date"]),
    axis=1
)

# Compute average values for synthetic data generation
mean_orders = summary_df["total_orders"].mean()
mean_spent = summary_df["total_spent"].mean()
mean_order_value = summary_df["avg_order_value"].mean()

# Calculate number of days since the last order
max_order_date = pd.to_datetime(orders_df["order_date"].max())
summary_df["days_since_last_order"] = (max_order_date - summary_df["last_order_date"]).dt.days
mean_days_since = summary_df["days_since_last_order"].mean()

# Generate 10,000 synthetic customer records with label=1 (reordered)
synthetic_data = []

for i in range(10000):
    total_orders = int(np.random.normal(loc=mean_orders + 2, scale=1))
    total_spent = np.random.normal(loc=mean_spent + 100, scale=50)
    avg_order_value = total_spent / total_orders
    last_order_date = pd.to_datetime("1997-10-01") + pd.Timedelta(days=random.randint(0, 30))
    days_since_last = (max_order_date - last_order_date).days

    synthetic_data.append({
        "customer_id": f"SYNTH{i}",
        "total_orders": total_orders,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "last_order_date": last_order_date,
        "days_since_last_order": days_since_last,
        "label": 1  # Synthetic data is assumed to reorder
    })

# Convert synthetic data to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Combine real and synthetic data
summary_df_augmented = pd.concat([summary_df, synthetic_df], ignore_index=True)

# Extract temporal features
summary_df_augmented["last_order_month"] = summary_df_augmented["last_order_date"].dt.month
summary_df_augmented["last_order_season"] = summary_df_augmented["last_order_month"].apply(
    lambda x: 0 if x in [12, 1, 2] else (1 if x in [3, 4, 5] else (2 if x in [6, 7, 8] else 3))
)

# Feature selection
features = ["total_spent", "total_orders", "avg_order_value", "days_since_last_order",
            "last_order_month", "last_order_season"]
X = summary_df_augmented[features]
y = summary_df_augmented["label"]

def train_and_save_model():
    """
    Train a neural network model to predict customer reordering behavior.
    Uses SMOTE to handle class imbalance and StandardScaler for normalization.
    Saves the trained model and scaler to disk.
    """
    df = summary_df_augmented

    # Re-extract date-based features for safety
    df['last_order_month'] = df['last_order_date'].dt.month
    df['last_order_season'] = df['last_order_month'].apply(
        lambda x: 0 if x in [12, 1, 2] else (1 if x in [3, 4, 5]
                                            else (2 if x in [6, 7, 8] else 3))
    )

    # Define features and target
    X = df[['total_spent', 'total_orders', 'avg_order_value', 'days_since_last_order', 'last_order_month', 'last_order_season']]
    y = df['label']

    # Scale features and balance data with SMOTE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Calculate class weights to handle imbalance
    class_weights = (
        dict(enumerate(compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)))
        if len(np.unique(y_train)) > 1 else None
    )

    # Build a simple feed-forward neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with early stopping
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[EarlyStopping(patience=5)], class_weight=class_weights)

    # Save the model and the scaler
    model_dir = "../saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "reorder_model.h5"))
    joblib.dump(scaler, os.path.join(model_dir, "reorder_scaler.pkl"))
    print("Model and scaler saved successfully.")

# Train the model and persist it
train_and_save_model()
