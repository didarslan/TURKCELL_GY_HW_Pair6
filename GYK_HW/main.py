from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

# 1) Önce app’i tanımlayın
app = Flask(__name__)



# 3) Model ve scaler’ı yükleyin
model  = tf.keras.models.load_model("customer_model.h5")
scaler = joblib.load("scaler.pkl")

# 4) Predict POST endpoint’i
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    cats = data["categories"]
    X = np.array([cats])
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]
    return jsonify({"purchase_probability": float(prob)})

if __name__ == "__main__":
    # Geliştirme için debug=True
    app.run(host="127.0.0.1", port=8000, debug=True)
