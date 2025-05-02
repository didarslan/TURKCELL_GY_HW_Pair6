# Turkcell GYK API Gateway

Bu proje, Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilen üç farklı ML API'sini tek bir API Gateway üzerinden erişilebilir hale getirir.

## Gereksinimler

- Python 3.10+
- FastAPI
- Uvicorn
- httpx

## Kurulum

1. Gerekli paketleri kurun:
```bash
cd api-gateway
pip install fastapi uvicorn httpx pydantic python-dotenv
```

## Çalıştırma

Tüm servisleri tek bir komutla başlatmak için:

```bash
cd api-gateway
debug_start.bat
```

Bu komut şunları yapacaktır:
1. Return Risk API'yi başlatır (port 8001)
2. Next Order API'yi başlatır (port 8002) 
3. New Product API'yi başlatır (port 8003)
4. API Gateway'i başlatır (port 8080)

## API Endpoints

Gateway üzerinden erişilebilen endpointler:

### Ürün İade Risk Skoru API
- `POST /return-risk/train`: Modeli eğitir
- `POST /return-risk/predict`: Manuel girdi ile tahmin yapar
- `POST /return-risk/predict_by_order`: Sipariş ID ile tahmin yapar

### Sonraki Sipariş Tahmini API
- `POST /next-order/predict`: Müşterinin yeni sipariş verme olasılığını tahmin eder

### Yeni Ürün Satın Alma Tahmini API
- `POST /new-product/predict`: Müşterinin yeni ürün satın alma olasılığını tahmin eder

## Örnek Kullanım

Swagger UI: `http://localhost:8080/docs`

API Gateway'in sağladığı Swagger dökümanları üzerinden tüm endpointleri test edebilirsiniz. 