# Ürün İade Risk Skoru API

Bu proje, müşterilerin sipariş verilerini kullanarak iade olasılığını tahmin eden bir derin öğrenme modeli ve API servisi sunar. Northwind veritabanındaki sipariş detaylarını analiz ederek, hangi siparişlerin iade edilme riski yüksek olduğunu belirler.

## Özet

İade tahmin modeli, dört ana özelliği kullanarak sipariş riskini değerlendirir:
- İndirim oranı
- Ürün miktarı
- Birim fiyat
- Toplam harcama tutarı

Model, özellikle yüksek indirim ve düşük harcama kombinasyonunun iade riski oluşturduğunu öğrenir. Cost-sensitive learning tekniği kullanılarak, iade olaylarına daha yüksek ağırlık verilir.

## Kullanılan Teknolojiler

- **Python 3.10+**
- **FastAPI**: Modern, hızlı web API framework'ü
- **TensorFlow/Keras**: Derin öğrenme modelinin oluşturulması ve eğitilmesi
- **scikit-learn**: Veri ön işleme ve model değerlendirme
- **PostgreSQL**: Northwind veritabanı için
- **psycopg2**: PostgreSQL bağlantısı
- **pandas & numpy**: Veri manipülasyonu ve analizi

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone [repository-url]
cd [proje-klasörü]
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. `.env` dosyası oluşturun ve veritabanı bilgilerinizi ekleyin:
```
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

## Veritabanı Hazırlığı

Proje, Northwind veritabanı üzerinde çalışmaktadır. Veritabanının kurulu olduğundan ve erişilebilir olduğundan emin olun. `config.py` dosyasında veritabanı yapılandırmasını gerektiği gibi değiştirebilirsiniz.

## API'yi Çalıştırma

```bash
python app.py
```

Uygulama varsayılan olarak `http://127.0.0.1:8000` adresinde çalışır. Swagger dokümantasyonuna `http://127.0.0.1:8000/docs` adresinden erişebilirsiniz.

## API Endpoint'leri

API üç ana endpoint sunar:

### 1. Modeli Eğitme

```
POST /train
```

Northwind veritabanından verileri çeker ve derin öğrenme modelini eğitir. Eğitim tamamlandığında model `return_risk_model.h5` dosyasına kaydedilir.

**Örnek Yanıt:**
```json
{
  "status": "başarılı",
  "message": "Model başarıyla eğitildi",
  "history": {
    "loss": [...],
    "accuracy": [...],
    "val_loss": [...],
    "val_accuracy": [...]
  }
}
```

### 2. Manuel Tahmin

```
POST /predict
```

Sipariş detaylarını kullanarak iade risk tahmini yapar.

**Örnek İstek:**
```json
{
  "discount": 0.25,
  "quantity": 10,
  "unit_price": 15,
  "total_amount": 112.5
}
```

**Örnek Yanıt:**
```json
{
  "status": "başarılı",
  "prediction": 0.78,
  "explanation": {
    "discount": 0.45,
    "quantity": 0.15,
    "unit_price": 0.15,
    "total_amount": 0.25
  }
}
```

### 3. Sipariş ID ile Tahmin

```
POST /predict_by_order
```

Veritabanından sipariş bilgilerini çekerek risk tahmini yapar.

**Örnek İstek:**
```json
{
  "order_id": 10248
}
```

**Örnek Yanıt:**
```json
{
  "status": "başarılı",
  "order_id": 10248,
  "items_count": 3,
  "items": [
    {
      "discount": 0.0,
      "quantity": 12,
      "unit_price": 14.0,
      "product_id": 11,
      "total_amount": 168.0,
      "prediction": 0.2,
      "explanation": {
        "discount": 0.2,
        "quantity": 0.2,
        "unit_price": 0.2,
        "total_amount": 0.4
      }
    },
    // ... diğer ürünler
  ],
  "average_risk_score": 0.3
}
```

## Tahmin Sonuçlarını Yorumlama

Risk skoru, 0 ile 1 arasında bir değerdir:

- **0.0 - 0.3**: Düşük risk - İade olasılığı düşük
- **0.3 - 0.6**: Orta risk - Orta seviye iade olasılığı
- **0.6 - 1.0**: Yüksek risk - Yüksek iade olasılığı

"explanation" alanı, hangi faktörlerin tahmini en çok etkilediğini gösterir. Örneğin, yüksek "discount" değeri, indirim oranının tahmin üzerinde büyük etkisi olduğunu belirtir.

## Model Mimarisi

Derin öğrenme modeli, 5 katmanlı bir sinir ağı kullanır:
- 128 nöronlu giriş katmanı (ReLU aktivasyon)
- 64 nöronlu gizli katman (ReLU aktivasyon)
- 32 nöronlu gizli katman (ReLU aktivasyon)
- 16 nöronlu gizli katman (ReLU aktivasyon)
- 1 nöronlu çıkış katmanı (Sigmoid aktivasyon)

Dropout katmanları (%10-%40) overfitting'i önlemek için kullanılmıştır.

## Ar-Ge Konuları

### Cost-Sensitive Learning

İade durumlarına 15 kat daha fazla ağırlık verilmiştir. Bu, modelin iade edilme olasılığı yüksek siparişleri daha iyi tanımasını sağlar. İade olayları daha nadir olduğundan ve firmaya daha fazla maliyeti olduğundan, bu asimetrik ağırlıklandırma önemlidir.

### Açıklanabilir AI (XAI)

Model tahminleri, kural tabanlı bir yöntemle açıklanmaktadır. Her bir özelliğin tahmine olan katkısı hesaplanır ve normalize edilir, böylece kullanıcılar neden bir siparişin yüksek risk taşıdığını anlayabilirler.

## İleri Düzey Kullanım

Modeli daha spesifik ihtiyaçlarınıza göre eğitebilirsiniz. Bunun için:

1. `fetch_data` metodunda SQL sorgusunu değiştirin
2. İş kurallarını `predict` metodunda düzenleyin
3. Derin öğrenme mimarisini `build_model` metodunda değiştirin

## Lisans

Bu proje akademik amaçlar için oluşturulmuştur. Ticari kullanım için izin gereklidir. 