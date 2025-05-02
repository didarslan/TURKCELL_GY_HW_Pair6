from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from model import ReturnRiskModel
from typing import Dict, Any, Optional, List, Union
import json

app = FastAPI(
    title="Ürün İade Risk Skoru API",
    description="Müşterilerin siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre iade riskini tahmin eden API",
    version="1.0.0"
)

model = ReturnRiskModel()

class PredictionRequest(BaseModel):
    discount: float  # İndirim oranı (0-1 arası)
    quantity: int    # Ürün miktarı
    unit_price: float  # Birim fiyat
    total_amount: float  # Toplam harcama miktarı

class OrderIdRequest(BaseModel):
    order_id: int  # Sipariş ID

class PredictionResponse(BaseModel):
    status: str
    prediction: float  # 0-1 arası (1'e yakın değerler yüksek iade riski)
    explanation: Dict[str, float]  # Her özelliğin tahmine katkısı

class OrderPredictionResponse(BaseModel):
    status: str
    order_id: int
    items_count: int
    items: List[Dict[str, Any]]
    average_risk_score: float

class TrainingResponse(BaseModel):
    status: str
    message: str
    history: Optional[Dict[str, List[float]]]

def convert_to_serializable(obj):
    """NumPy tiplerini Python tiplerine dönüştür"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

@app.post("/train", response_model=TrainingResponse, 
         description="Modeli eğitir. Cost-sensitive learning kullanarak iade durumlarına daha yüksek ağırlık verir.")
async def train_model():
    """
    Derin öğrenme modelini eğitir.
    
    - Cost-sensitive learning kullanır (iade durumları 5 kat daha ağırlıklı)
    - Eğitim tamamlandığında model dosyaya kaydedilir
    
    Returns:
        Eğitim durumu ve metrikleri
    """
    try:
        history = model.train_and_save()
        
        # NumPy tiplerini Python tiplerine dönüştür
        serializable_history = {}
        if hasattr(history, 'history'):
            serializable_history = convert_to_serializable(history.history)
        
        return {
            "status": "başarılı",
            "message": "Model başarıyla eğitildi",
            "history": serializable_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse,
         description="Manuel girilen sipariş bilgilerine göre iade risk tahmini yapar.")
async def predict(request: PredictionRequest):
    """
    Bir siparişin iade edilme riskini tahmin eder.
    
    - 0-1 arası bir risk skoru döndürür (1'e yakın değerler = yüksek risk)
    - Model özellik önemleri ile hangi faktörlerin tahmine en çok etki ettiğini açıklar
    
    Args:
        request: İndirim oranı, miktar, birim fiyat ve toplam tutar bilgileri
        
    Returns:
        Tahmin sonucu ve açıklaması
    """
    try:
        # Girdi verilerini hazırla
        input_data = request.dict()
        
        # Tahmin yap
        prediction = model.predict(input_data)
        
        # Tahmini açıkla
        explanation = model.explain_prediction(input_data)
        
        return {
            "status": "başarılı",
            "prediction": float(prediction),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_by_order", response_model=OrderPredictionResponse,
         description="Sipariş ID'sine göre veritabanından sipariş bilgilerini çekip iade risk tahmini yapar.")
async def predict_by_order(request: OrderIdRequest):
    """
    Sipariş ID'si kullanarak bir siparişin iade edilme riskini tahmin eder.
    
    - Veritabanından sipariş bilgilerini çeker
    - Siparişin her bir ürünü için risk tahmini yapar
    - Ortalama risk skorunu hesaplar
    
    Args:
        request: Sipariş ID
        
    Returns:
        Siparişin tüm ürünleri için tahmin sonuçları ve ortalama risk skoru
    """
    try:
        # Sipariş verileri ile tahmin yap
        result = model.predict_by_order_id(request.order_id)
        
        return {
            "status": "başarılı",
            **result
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Swagger UI'da ek açıklamalar
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 