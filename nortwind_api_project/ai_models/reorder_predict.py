# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from datetime import timedelta
from imblearn.over_sampling import SMOTE
import random
import os
import joblib

# PostgreSQL veritabanı bağlantısı
db_url = "postgresql://postgres:***@localhost:5432/***"
engine = create_engine(db_url)

# SQL sorguları: Müşteri bazlı özet ve sipariş verileri
summary_query = '''
SELECT 
  c.customer_id AS customer_id,
  COUNT(DISTINCT o.order_id) AS total_orders,
  SUM(od.unit_price * od.quantity) AS total_spent,
  AVG(od.unit_price * od.quantity) AS avg_order_value,
  MAX(o.order_date) AS last_order_date
FROM Customers c
JOIN Orders o ON c.customer_id = o.customer_id
JOIN Order_Details od ON o.order_id = od.order_id
GROUP BY c.customer_id;
'''

orders_query = '''
SELECT customer_id, order_date FROM Orders;
'''

# SQL'den verilerin çekilmesi
summary_df = pd.read_sql(summary_query, engine)
orders_df = pd.read_sql(orders_query, engine)

# Tarih sütunlarının datetime formatına dönüştürülmesi
summary_df['last_order_date'] = pd.to_datetime(summary_df['last_order_date'])
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

# Etiket oluşturma: 6 ay içinde tekrar sipariş var mı?
def has_order_in_next_6_months(customer_id, last_date):
    """
        Belirli bir müşterinin son siparişinden sonraki 6 ay içinde yeni sipariş verip vermediğini kontrol eder.

        Args:
            customer_id (str): Müşteri ID'si
            last_date (datetime): Müşterinin son sipariş tarihi

        Returns:
            int: 1 (sipariş var), 0 (yok)
        """
    future_orders = orders_df[
        (orders_df["customer_id"] == customer_id) &
        (orders_df["order_date"] > last_date) &
        (orders_df["order_date"] <= last_date + pd.Timedelta(days=180))
    ]
    return 1 if not future_orders.empty else 0

# Etiket değişkeni oluşturuldu
summary_df["label"] = summary_df.apply(
    lambda row: has_order_in_next_6_months(row["customer_id"], row["last_order_date"]),
    axis=1
)

# Temel istatistikler
mean_orders = summary_df["total_orders"].mean()
mean_spent = summary_df["total_spent"].mean()
mean_order_value = summary_df["avg_order_value"].mean()

max_order_date = pd.to_datetime(orders_df["order_date"].max())
summary_df["days_since_last_order"] = (max_order_date - summary_df["last_order_date"]).dt.days
mean_days_since = summary_df["days_since_last_order"].mean()

# Veri artırımı: 10.000 adet yapay müşteri verisi oluşturuldu
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
        "label": 1  # Yapay veriler için etiket 1
    })

synthetic_df = pd.DataFrame(synthetic_data)

# Orijinal ve yapay verilerin birleştirilmesi
summary_df_augmented = pd.concat([summary_df, synthetic_df], ignore_index=True)

# Yeni temporal özellikler (Ay & Mevsim etkisi)
summary_df_augmented["last_order_month"] = summary_df_augmented["last_order_date"].dt.month
summary_df_augmented["last_order_season"] = summary_df_augmented["last_order_month"].apply(
    lambda x: 0 if x in [12, 1, 2] else (1 if x in [3, 4, 5] else (2 if x in [6, 7, 8] else 3))
)

# Özellikler ve etiketlerin ayrılması
features = ["total_spent", "total_orders", "avg_order_value", "days_since_last_order",
            "last_order_month", "last_order_season"]
X = summary_df_augmented[features]
y = summary_df_augmented["label"]

def train_and_save_model():
    """
    Müşteri tekrar sipariş tahmini için bir yapay sinir ağı modeli eğitir,
    veriyi ölçeklendirir, SMOTE ile dengeler ve eğitilen modeli disk'e kaydeder.
    """
    df = summary_df_augmented

    # Tarih temelli özelliklerin tekrar çıkarılması (güvenlik amaçlı)
    df['last_order_month'] = df['last_order_date'].dt.month
    df['last_order_season'] = df['last_order_month'].apply(
        lambda x: 0 if x in [12, 1, 2] else (1 if x in [3, 4, 5]
                                            else (2 if x in [6, 7, 8] else 3))
    )

    # Özellikler ve etiket
    X = df[['total_spent', 'total_orders', 'avg_order_value', 'days_since_last_order', 'last_order_month', 'last_order_season']]
    y = df['label']

    # Veriyi ölçeklendirme ve SMOTE ile dengeleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    # Eğitim ve test verisi ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Sınıf ağırlıklarını hesaplama
    class_weights = (
        dict(enumerate(compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)))
        if len(np.unique(y_train)) > 1 else None
    )

    # Modeli oluşturma
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Modeli eğitme
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[EarlyStopping(patience=5)], class_weight=class_weights)

    # Model ve scaler'ı kaydetme
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "reorder_model.h5"))
    joblib.dump(scaler, os.path.join(model_dir, "reorder_scaler.pkl"))
    print("Model ve scaler kaydedildi.")

train_and_save_model()
