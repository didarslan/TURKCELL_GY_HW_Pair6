 Müşteri Satın Alma Tahmini - Yapay Zeka Projesi
Bu proje, Northwind veritabanındaki geçmiş sipariş verilerine bakarak bir müşterinin yeni bir ürünü (örneğin Beverages) alma olasılığını tahmin eden bir yapay sinir ağı modelini içerir. Model, Flask tabanlı bir REST API aracılığıyla servis edilmektedir.

 ##Proje Dosyaları

train_model.py → Veriyi veritabanından çeker, işler ve modeli eğitir.

scaler.pkl → Girdi verilerini normalize etmek için kullanılan MinMaxScaler.

customer_model.h5 → Eğitilmiş Keras modelidir. Flask API tarafından yüklenir.

main.py → Flask tabanlı API. /predict endpoint’i ile tahmin yapılmasını sağlar.

app.py → (Opsiyonel) Alternatif bir API dosyası. Kullanılmıyorsa silinebilir.

##Gereksinimler ve Kurulum

Aşağıdaki Python kütüphanelerinin kurulu olması gerekir:

pip install flask pandas numpy tensorflow scikit-learn sqlalchemy psycopg2-binary joblib

##Modeli Eğitme

Aşağıdaki komut ile modeli eğitebilirsiniz:

python train_model.py

Bu işlem sonucunda iki dosya oluşur: customer_model.h5 (model) ve scaler.pkl (ölçekleyici).

##Flask API'yi Başlatma

python main.py komutunu çalıştırın. Ardından API şu adreste aktif olur:
http://127.0.0.1:8000/predict

##Örnek API İsteği

Endpoint: /predict
Method: POST
Content-Type: application/json

İstek örneği:
{
"categories": [500, 200, 0, 0, 0, 0, 0, 120]
}

Bu örnek, bir müşterinin bazı ürün kategorilerine ne kadar harcama yaptığını temsil eder. Model bu girdilere göre Beverages kategorisinden bir ürün alma ihtimalini hesaplar.
