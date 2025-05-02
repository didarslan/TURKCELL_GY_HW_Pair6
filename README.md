# Turkcell GYK - Bütünleştirilmiş API Servisleri

Bu repoda, Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilen üç bağımsız ML API projesini birleştiren bir API Gateway bulunmaktadır. Bu proje, farklı adreslerdeki ML servislerini tek bir endpoint altında birleştirerek, tüm servislerin merkezi bir şekilde yönetilmesini sağlar.

## Projeler

Bu repo içerisinde üç farklı ML API projesi bulunmaktadır:

### 1. Return Risk Predictor (Ürün İade Risk Tahmini)

**Klasör:** `return-risk-predictor`

**Açıklama:** Müşterilerin sipariş verilerini analiz ederek iade edilme olasılığı yüksek olan siparişleri tahmin eden bir derin öğrenme modeli ve API sunar.

**Endpointler:**
- `POST /return-risk/train`: Modeli eğitir
- `POST /return-risk/predict`: Manuel girdi ile tahmin yapar
- `POST /return-risk/predict_by_order`: Sipariş ID ile tahmin yapar

### 2. Next Order Predictor (Sonraki Sipariş Tahmini)

**Klasör:** `next-order-predictor`

**Açıklama:** Müşterilerin geçmiş sipariş davranışlarını analiz ederek yakın zamanda yeni bir sipariş verme olasılığını tahmin eden bir derin öğrenme modeli ve API içerir.

**Endpointler:**
- `POST /next-order/predict`: Müşterinin yeni sipariş verme olasılığını tahmin eder

### 3. New Product Purchase Predictor (Yeni Ürün Satın Alma Tahmini)

**Klasör:** `new-product-purchase-predictor`

**Açıklama:** Müşterinin ilgi duyduğu ürün kategorilerine göre yeni bir ürün satın alma olasılığını tahmin eden bir derin öğrenme modeli ve API sunar.

**Endpointler:**
- `POST /new-product/predict`: Müşterinin yeni ürün satın alma olasılığını tahmin eder

## API Gateway Entegrasyonu

**Klasör:** `api-gateway`

API Gateway, yukarıdaki üç servisi birleştirerek tek bir API üzerinden erişilebilir hale getirir. Bu sayede farklı servislere ayrı ayrı istek yapmak yerine, tek bir endpoint üzerinden tüm servislere erişilebilir.

### Gereksinimler

- Python 3.10+
- FastAPI
- Uvicorn
- httpx

### Kurulum

1. Gerekli paketleri yükleyin:
```bash
cd api-gateway
pip install fastapi uvicorn httpx pydantic python-dotenv
```

Ayrıca, her bir servisin çalışması için ilgili klasörlerdeki requirements.txt dosyalarını yüklemeniz gerekebilir.

### Çalıştırma

Tüm servisleri ve API Gateway'i tek bir komutla başlatmak için:

```bash
cd api-gateway
debug_start.bat
```

Bu komut sırasıyla:
1. Return Risk API'yi başlatır (port 8001)
2. Next Order API'yi başlatır (port 8002) 
3. New Product API'yi başlatır (port 8003)
4. API Gateway'i başlatır (port 8080)

### Kullanım

API Gateway'e erişmek için: `http://localhost:8080/docs`

Swagger UI üzerinden tüm endpointleri test edebilirsiniz. API Gateway, isteği uygun servise yönlendirir ve yanıtı alıp size geri döndürür.

## Örnek İstekler

### İade Risk Tahmini

```json
POST /return-risk/predict
{
  "discount": 0.1,
  "quantity": 5,
  "unit_price": 29.99,
  "total_amount": 134.95
}
```

### Sonraki Sipariş Tahmini

```json
POST /next-order/predict
{
  "total_spent": 500.0,
  "total_orders": 10,
  "avg_order_value": 50.0,
  "days_since_last_order": 30,
  "last_order_month": 3,
  "last_order_season": 1
}
```

### Yeni Ürün Satın Alma Tahmini

```json
POST /new-product/predict
{
  "categories": [0, 1, 0, 0, 1, 0, 1, 0]
}
```

## Mimari

```
                    ┌───────────────────┐
                    │                   │
                    │   API Gateway     │
                    │   (port 8080)     │
                    │                   │
                    └─────────┬─────────┘
                              │
           ┌─────────────────┼────────────────┐
           │                 │                │
  ┌────────▼─────────┐ ┌─────▼──────────┐ ┌───▼───────────────┐
  │                  │ │                │ │                   │
  │  Return Risk     │ │  Next Order    │ │  New Product      │
  │  Predictor       │ │  Predictor     │ │  Purchase         │
  │  (port 8001)     │ │  (port 8002)   │ │  Predictor        │
  │                  │ │                │ │  (port 8003)      │
  └──────────────────┘ └────────────────┘ └───────────────────┘
```

## Katkıda Bulunanlar

Bu proje Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilmiştir.

| İsim           | 
|----------------|
| Elif Barutçu   |
| Özge Taraşlı   | 
| Mine Emektar   | 
| Didar Arslan   |
| Deniz Tunç     |
| Nurefşan Gültekin |
