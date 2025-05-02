
# TURKCELL_GY_HW_Pair6 - Proje Kataloğu

Bu repo, Turkcell Geleceği Yazan Kadınlar Programı kapsamında geliştirilen üç bağımsız projenin bir araya getirildiği birleşik bir çalışmadır. Her proje, veri bilimi ve yapay zeka uygulamalarını gerçek dünya problemlerine çözüm olacak şekilde ele alır.

---

## 📁 1. Ürün İade Risk Skoru API

**Klasör:** [`return-risk-predictor`](./return-risk-predictor)  

Bu proje, müşterilerin sipariş verilerini analiz ederek iade edilme olasılığı yüksek olan siparişleri tahmin eden bir derin öğrenme modeli ve API sunar. Northwind veritabanı üzerinden alınan sipariş detayları ile model, iade riski yüksek siparişleri tanımlar.

### ⚙️ Kullanılan Teknolojiler
- Python 3.10+
- FastAPI
- TensorFlow / Keras
- scikit-learn
- PostgreSQL
- psycopg2
- pandas, numpy

### 🔍 Model Özellikleri
- Giriş değişkenleri: `discount`, `quantity`, `unit_price`, `total_amount`
- Cost-sensitive learning: iade sınıfına 15x ağırlık
- 5 katmanlı derin sinir ağı + dropout
- Açıklanabilir tahminler (`explanation` alanı)

### 📌 API Endpoint'leri
- `POST /train`: Modeli eğitir
- `POST /predict`: Manuel girdi ile tahmin
- `POST /predict_by_order`: Sipariş ID ile tahmin

### 📊 Tahmin Sonuçlarının Yorumlanması
| Risk Skoru | Açıklama            |
|------------|---------------------|
| 0.0 - 0.3  | Düşük risk          |
| 0.3 - 0.6  | Orta risk           |
| 0.6 - 1.0  | Yüksek iade riski   |

Detaylı kullanım için klasör içindeki `README.md` ve `docs` dizinine bakabilirsiniz.

---

## 📁 2. GYK_HW - Eğitim Uygulamaları

**Klasör:** [`GYK_HW`](./GYK_HW)  

Bu klasör, eğitim boyunca yapılan ödev ve uygulamaları içerir. Katılımcı, veri analizi ve temel makine öğrenmesi konularında geliştirdiği çalışmalarla Python, pandas, seaborn gibi araçlarla çeşitli senaryoları analiz etmiştir.

### 🔍 İçerikler
- Veri temizleme ve görselleştirme örnekleri
- Pandas pivot, merge işlemleri
- Modelleme örnekleri
- Jupyter notebook tabanlı projeler

---

## 📁 3. Northwind Ürün Öneri API'si

**Klasör:** [`nortwind_api_project`](./nortwind_api_project)  

Bu proje, Northwind veri seti üzerinde çalışan makine öğrenmesi tabanlı bir ürün öneri sistemidir. Kullanıcının geçmiş satın alma verileri analiz edilerek benzer ürünler önerilir. API, FastAPI framework'ü ile servisleştirilmiştir.

### 🧠 Kullanılan Yöntemler
- KMeans tabanlı segmentasyon
- Kullanıcıya özel öneri mantığı
- Model eğitimi ve tahmin endpoint'leri

### 📌 API Özellikleri
- `/train`: modeli eğitir
- `/recommend`: kullanıcıya ürün önerir
- PostgreSQL veritabanına bağlanarak Northwind verisini kullanır

---

## 🧪 Kurulum ve Çalıştırma

1. Reposu klonlayın:
```bash
git clone https://github.com/didarslan/TURKCELL_GY_HW_Pair6.git
cd TURKCELL_GY_HW_Pair6
```

2. Her klasör için:
```bash
cd [klasör_adı]
pip install -r requirements.txt
```

3. `.env` dosyası oluşturarak veritabanı bilgilerinizi girin:
```
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

4. API'leri başlatmak için:
```bash
uvicorn app:app --reload
# veya
python app.py
```

---

## 👥 Katkıda Bulunanlar

| İsim           | 
|----------------|
| Elif Barutçu   |
| Özge Taraşlı   | 
| Mine Emektar   | 
| Didar Arslan   |
| Deniz Tunç     |
| Nurefşan Gültekin |

---

> Bu çalışma, Turkcell Geleceği Yazan Kadınlar 2025 programı kapsamında hazırlanmıştır.
