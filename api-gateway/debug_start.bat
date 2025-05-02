@echo off
echo Tüm API servisleri başlatılıyor...

echo 1. Return Risk API başlatılıyor (port 8001)...
start cmd /k "cd ../return-risk-predictor && python -m uvicorn app:app --host 0.0.0.0 --port 8001"

echo 2. Next Order API başlatılıyor (port 8002)...
start cmd /k "cd ../next-order-predictor && python -m uvicorn app:app --host 0.0.0.0 --port 8002"

echo 3. New Product API başlatılıyor (port 8003)...
start cmd /k "cd ../new-product-purchase-predictor && python -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8003"

echo 4. API Gateway başlatılıyor (port 8080)...
cd %~dp0
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload

echo Tüm servisler başlatıldı!
echo API Gateway: http://localhost:8080/docs 