import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sqlalchemy import create_engine



#PostgreSQL veritabanına bağlanmak için gerekli temel bilgiler.
#Bu bilgiler yerel bir Northwind veritabanına erişim sağlıyor.


pg_host     = "localhost"   # Buradaki Host name/address
pg_port     = "5433"        # Buradaki Port
pg_db       = "Gyk1Northwind"   # Buradaki Maintenance DB
pg_user     = "postgres"    # Buradaki Username
pg_password = "2001"   # Buradaki Password


#Bu kod parçası PostgreSQL veritabanına bağlanmak için SQLAlchemy kütüphanesini kullanır.
#SQLAlchemy, Python’dan veritabanı yönetim sistemlerine (ör. PostgreSQL, MySQL, SQLite...) bağlanmak için kullanılır. 
#Bu bağlantı kurulduktan sonra veritabanından SQL sorgusu ile veri çekebilirsin.

# SQLAlchemy bağlantı URL’si
conn_str = (
    f"postgresql+psycopg2://{pg_user}:{pg_password}"
    f"@{pg_host}:{pg_port}/{pg_db}"
)

# Engine oluştur
engine = create_engine(conn_str)

# Çekmek istediğiniz sorgu
sql = """
SELECT o.customer_id, c.category_name, SUM(od.unit_price * od.quantity) AS TotalSpent
FROM public.orders o
JOIN public.order_details od ON o.order_id = od.order_id
JOIN public.products p ON od.product_id = p.product_id
JOIN public.categories c ON p.category_id = c.category_id
GROUP BY o.customer_id, c.category_name
ORDER BY o.customer_id;

"""

#  SQL sorgusundan gelen verileri pandas DataFrame’ine yükler.
df = pd.read_sql_query(sql, engine)

# Sonuçları görüntüle
#print(df.head())

# Sütun isimlerini küçük harfe çevirip gereksiz boşlukları siler.
df.columns = df.columns.str.lower().str.strip()

# 2-Pivot
#Her satır bir müşteri olur
#Her sütun bir kategori olur
#Hücrelerde toplam harcama yer alır
#fillna(0) ile boş (harcama yapılmamış) yerler sıfır olur


customer_features = (
    df
    .pivot(
      index='customer_id',
      columns='category_name',
      values='totalspent'
    )
    .fillna(0)
)

# 3-Bu kısım, modelin tahmin edeceği etiketi (y) üretir. (örnek: Beverages almış mı?)
customer_features['target_beverages'] = np.where(
    customer_features['Beverages'] > 0,
    1,
    0
)

# 4- X, y olarak ayırın
#X: Girdi verileri (müşterinin farklı kategorilere yaptığı harcamalar)
#y: Çıktı (hedef) — yani bu müşteri Beverages kategorisinden ürün almış mı?

X = customer_features.drop('target_beverages', axis=1)
y = customer_features['target_beverages']

#print(df.columns.tolist())
#print(type(customer_features['Beverages']))            # bu bir Series olmalı
#print(type(customer_features['Beverages'] > 0))       # bu da bir Series olmalı
#print(customer_features['Beverages'] > 0)             # burada bir tablo görmelisiniz, tek bir True/False değil



# Veriyi normalize edelim:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # İhtimal tahmini için sigmoid
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Değerlendir modeli test et
model.evaluate(X_test, y_test)

cats = X.columns.tolist()

# Tüm sütunlar için önce 0’dan oluşan bir sözlük oluşturma. 
# cats, modelde kullanılan tüm kategori isimlerinin listesidir (örneğin: 'Beverages', 'Seafood', 'Confections' vs).
template = dict.fromkeys(cats, 0)

# İlgilendiğimiz kategorilere değer atama
template['Beverages']   = 500
template['Confections'] = 200
template['Seafood']     = 120
# diğer kategoriler zaten 0

# Tek satırlık bir DataFrame’e dönüştürme
new_df = pd.DataFrame([template], columns=cats)

# Ölçekleyiciyle dönüştürme
# Eğitimde kullanılan MinMaxScaler, 
# yeni müşterinin verilerine de uygulanmalı ki ölçekler modelle uyumlu olsun.
#Bu sayede model, bu yeni veriyi tanıyabileceği bir forma sokar.
new_customer_scaled = scaler.transform(new_df)

# Tahmin yapıp sonucu bastırma
probability = model.predict(new_customer_scaled)
print(f"Yeni ürünü alma ihtimali: {probability[0][0]:.2%}")


#api haline getirmek için

model.save('customer_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')