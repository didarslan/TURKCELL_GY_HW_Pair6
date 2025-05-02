from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from config import (
    RETURN_RISK_API_URL,
    NEXT_ORDER_API_URL,
    NEW_PRODUCT_API_URL
)
from typing import Dict, Any, List, Optional

app = FastAPI(
    title="Turkcell GYK API Gateway",
    description="Turkcell Geleceği Yazan Kadınlar Programı projelerini birleştiren API Gateway",
    version="1.0.0"
)

# Model tanımları
# Return Risk API için modeller
class PredictionRequest(BaseModel):
    discount: float
    quantity: int
    unit_price: float
    total_amount: float

class OrderIdRequest(BaseModel):
    order_id: int

# Next Order API için modeller
class CustomerFeatures(BaseModel):
    total_spent: float
    total_orders: int
    avg_order_value: float
    days_since_last_order: int
    last_order_month: int
    last_order_season: int

# New Product API için modeller
class ProductPredictionRequest(BaseModel):
    categories: List[float]

# API Gateway ana sayfası
@app.get("/")
async def root():
    return {
        "message": "Turkcell GYK API Gateway",
        "services": [
            {"name": "Ürün İade Risk Skoru API", "base_url": "/return-risk"},
            {"name": "Sonraki Sipariş Tahmini API", "base_url": "/next-order"},
            {"name": "Yeni Ürün Satın Alma Tahmini API", "base_url": "/new-product"}
        ]
    }

#----- Ürün İade Risk Skoru API Endpoints -----
@app.post("/return-risk/train")
async def train_return_risk_model():
    """Ürün iade risk skorlama modelini eğitir"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{RETURN_RISK_API_URL}/train")
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Servis hatası: {str(exc)}")

@app.post("/return-risk/predict")
async def predict_return_risk(request: PredictionRequest):
    """Ürün iade risk skorunu tahmin eder"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RETURN_RISK_API_URL}/predict", 
                json=request.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Servis hatası: {str(exc)}")

@app.post("/return-risk/predict_by_order")
async def predict_by_order_id(request: OrderIdRequest):
    """Sipariş ID'sine göre iade risk skorunu tahmin eder"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RETURN_RISK_API_URL}/predict_by_order", 
                json=request.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Servis hatası: {str(exc)}")

#----- Sonraki Sipariş Tahmini API Endpoints -----
@app.post("/next-order/predict")
async def predict_next_order(customer: CustomerFeatures):
    """Müşterinin yeni sipariş verme olasılığını tahmin eder"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{NEXT_ORDER_API_URL}/reorder_predict", 
                json=customer.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Servis hatası: {str(exc)}")

#----- Yeni Ürün Satın Alma Tahmini API Endpoints -----
@app.post("/new-product/predict")
async def predict_product_purchase(request: ProductPredictionRequest):
    """Müşterinin yeni ürün satın alma olasılığını tahmin eder"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{NEW_PRODUCT_API_URL}/predict", 
                json=request.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Servis hatası: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    from config import API_GATEWAY_HOST, API_GATEWAY_PORT
    
    uvicorn.run(app, host=API_GATEWAY_HOST, port=API_GATEWAY_PORT) 