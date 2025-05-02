# Turkcell Geleceği Yazan Kadınlar - Müşteri Davranış Tahmin Servisleri

Bu proje, Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilmiş olan, müşteri davranışlarını makine öğrenmesi yöntemleriyle tahmin eden üç farklı mikroservisi ve bunları birleştiren bir API Gateway içerir.

## Proje Yapısı

Proje, aşağıdaki dört ana bileşenden oluşmaktadır:

### 1. API Gateway

**Klasör:** `api-gateway`

**Açıklama:** Diğer üç ML servisini tek bir API üzerinden erişilebilir hale getiren, istek yönlendirme ve ortak arayüz sağlayan servis. FastAPI ile geliştirilmiştir.

**Anahtar Özellikler:**
- Tüm servisleri tek bir endpoint üzerinden sunar
- Swagger dokümantasyonu ile kolay test imkanı
- Asenkron HTTP istekleri ile yüksek performans

### 2. Ürün İade Risk Tahmini (Return Risk Predictor)

**Klasör:** `return-risk-predictor`

**Açıklama:** Müşterilerin sipariş verilerini analiz ederek siparişin iade edilme olasılığını tahmin eden derin öğrenme modeli ve API.

**Anahtar Özellikler:**
- İndirim oranı, ürün miktarı, birim fiyat gibi faktörlere göre tahmin
- Tahmin sonucunu açıklayan feature importance analizi
- Sipariş ID'sine göre sorgulama imkanı

### 3. Sonraki Sipariş Tahmini (Next Order Predictor)

**Klasör:** `next-order-predictor`

**Açıklama:** Müşterilerin geçmiş sipariş davranışlarını analiz ederek yakın zamanda yeni bir sipariş verme olasılığını tahmin eden derin öğrenme modeli ve API.

**Anahtar Özellikler:**
- Toplam harcama, sipariş sayısı, ortalama sipariş değeri gibi metriklere dayalı tahmin
- Son siparişten bu yana geçen zaman analizi
- Mevsimsellik faktörlerini dikkate alan model

### 4. Yeni Ürün Satın Alma Tahmini (New Product Purchase Predictor)

**Klasör:** `new-product-purchase-predictor`

**Açıklama:** Müşterinin ilgi duyduğu ürün kategorilerine göre yeni bir ürün satın alma olasılığını tahmin eden derin öğrenme modeli ve API.

**Anahtar Özellikler:**
- Kategori bazlı müşteri tercihlerini analiz eder
- Yeni ürün satın alma olasılığını tahmin eder

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.10+
- FastAPI
- Uvicorn
- httpx
- TensorFlow
- pandas, numpy, scikit-learn

### Kurulum Adımları

1. Repo'yu klonlayın:
```bash
git clone <repo-url>
cd <repo-folder>
```

2. Her bir servis için gerekli paketleri yükleyin:
```bash
cd api-gateway
pip install -r requirements.txt

cd ../return-risk-predictor
pip install -r requirements.txt

cd ../next-order-predictor
pip install -r requirements.txt

cd ../new-product-purchase-predictor
pip install -r requirements.txt
```

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

## API Kullanımı

API Gateway'e erişmek için: `http://localhost:8080/docs`

Bu adreste Swagger UI aracılığıyla tüm API endpointlerini inceleyebilir ve test edebilirsiniz.

### Servis Endpointleri

**Ürün İade Risk Skoru API**
- `POST /return-risk/train`: Modeli eğitir
- `POST /return-risk/predict`: Manuel girdi ile tahmin yapar
- `POST /return-risk/predict_by_order`: Sipariş ID ile tahmin yapar

**Sonraki Sipariş Tahmini API**
- `POST /next-order/predict`: Müşterinin yeni sipariş verme olasılığını tahmin eder

**Yeni Ürün Satın Alma Tahmini API**
- `POST /new-product/predict`: Müşterinin yeni ürün satın alma olasılığını tahmin eder

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

## Mimari Diyagramı

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

## Teknolojiler

- **Backend Framework:** FastAPI
- **ML Framework:** TensorFlow, scikit-learn
- **Veri İşleme:** pandas, numpy
- **API Client:** httpx
- **API Dokümantasyonu:** Swagger UI (FastAPI entegrasyonu)

## Katkıda Bulunanlar

Bu proje Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilmiştir.

| İsim               |
|--------------------|
| Elif Barutçu       |
| Özge Taraşlı       |
| Mine Emektar       |
| Didar Arslan       |
| Deniz Tunç         |
| Nurefşan Gültekin  |
