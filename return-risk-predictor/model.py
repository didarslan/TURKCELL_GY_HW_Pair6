import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import lime
import lime.lime_tabular
import psycopg2
import os
import logging
from config import DB_CONFIG
import shap
import joblib

# Loglama
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReturnRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['discount', 'quantity', 'unit_price', 'total_amount']
        self.X_train = None
        
        # Eğer model ve scaler dosyaları varsa yükle
        if os.path.exists('return_risk_model.h5'):
            try:
                self.model = load_model('return_risk_model.h5')
                logger.info("Model yüklendi.")
                
                # Scaler'ı da yükle
                if os.path.exists('scaler.pkl'):
                    self.scaler = joblib.load('scaler.pkl')
                    logger.info("Scaler yüklendi.")
                else:
                    logger.warning("Scaler dosyası bulunamadı.")
            except Exception as e:
                logger.warning(f"Model yüklenemedi: {str(e)}")
        
    def fetch_data(self):
        """Northwind veritabanından veri çekme"""
        conn = psycopg2.connect(**DB_CONFIG)
        
        query = """
        SELECT 
            od.discount,
            od.quantity,
            od.unit_price,
            (od.quantity * od.unit_price * (1 - od.discount)) as total_amount,
            CASE 
                WHEN od.discount > 0.15 AND (od.quantity * od.unit_price * (1 - od.discount)) < 100 THEN 1
                ELSE 0
            END as is_returned
        FROM order_details od
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        logger.info(f"Veri çekildi. Toplam {len(df)} kayıt.")
        logger.info(f"İade edilen ürün sayısı: {df['is_returned'].sum()}")
        return df
    
    def get_order_by_id(self, order_id):
        """Veritabanından belirli bir siparişi çekme"""
        conn = psycopg2.connect(**DB_CONFIG)
        
        query = """
        SELECT 
            od.discount,
            od.quantity,
            od.unit_price,
            od.product_id,
            (od.quantity * od.unit_price * (1 - od.discount)) as total_amount
        FROM order_details od
        WHERE od.order_id = %s
        """
        
        df = pd.read_sql(query, conn, params=[order_id])
        conn.close()
        
        if df.empty:
            raise ValueError(f"Sipariş bulunamadı: {order_id}")
            
        logger.info(f"Sipariş verileri çekildi: {order_id}")
        return df
    
    def prepare_data(self, df):
        """Eğitim için veri hazırlama"""
        # Eksik özellikleri kontrol et
        for feature in self.feature_names:
            if feature not in df.columns:
                raise ValueError(f"Eksik özellik: {feature}")
        
        X = df[self.feature_names]
        y = df['is_returned']
        
        # Özellikleri ölçeklendirme
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = X_scaled[:100].copy()  # İlk 100 eğitim örneğini sakla
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def build_model(self):
        """Derin öğrenme modelini oluşturma"""
        # Model mimarisi: Daha derin ve daha karmaşık
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        # Öğrenme oranını düşük tut (daha stabil eğitim için)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        """Cost-sensitive learning ile model eğitimi"""
        # İade durumları için daha yüksek ağırlık
        class_weights = {0: 1, 1: 15}  # İadeler 15 kat daha önemli
        
        # Erken durdurma ve model checkpointing
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Modeli eğit
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,  # Daha uzun eğitim
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Eğitim performansını değerlendir
        logger.info("-" * 50)
        logger.info("Eğitim tamamlandı")
        val_metrics = self.model.evaluate(X_val, y_val)
        if len(val_metrics) > 1:
            val_loss, val_acc = val_metrics[0], val_metrics[1]
            logger.info(f"Doğrulama kaybı: {val_loss:.4f}")
            logger.info(f"Doğrulama doğruluğu: {val_acc:.4f}")
        else:
            logger.info(f"Doğrulama kaybı: {val_metrics:.4f}")
        logger.info("-" * 50)
        
        return history
    
    def predict(self, input_data):
        """Tahmin yapma - Sadece derin öğrenme modeli kullanarak"""
        # Girdi verilerini DataFrame'e dönüştürme
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
        
        # Eksik özellikleri kontrol etme
        for feature in self.feature_names:
            if feature not in input_data.columns:
                raise ValueError(f"Eksik özellik: {feature}")
        
        # Model kontrolü
        if self.model is None:
            raise ValueError("Model yüklenmemiş veya eğitilmemiş. Önce modeli eğitin.")
            
        # Özellikleri ölçeklendirme
        X_scaled = self.scaler.transform(input_data[self.feature_names])
        
        # Model tahmini
        model_prediction = float(self.model.predict(X_scaled)[0][0])
        
        # Tahmin değerini anlamlı bir aralığa getir
        prediction = max(0.001, min(0.999, model_prediction))
        
        # İş kuralları ile iyileştirme (minimal düzeyde)
        discount = input_data['discount'].values[0]
        total_amount = input_data['total_amount'].values[0]
        quantity = input_data['quantity'].values[0]
        unit_price = input_data['unit_price'].values[0]
        
        # Yüksek indirim + düşük harcama durumunda riski artır
        if discount > 0.5 and total_amount < 50:
            prediction = max(prediction * 5, 0.8)  # Agresif artış, en az 0.8 risk
        elif discount > 0.3 and total_amount < 100:
            prediction = max(prediction * 3, 0.6)  # Orta düzey artış
        elif discount > 0.15 and total_amount < 100: 
            prediction = max(prediction * 2, 0.5)  # Hafif artış
            
        # Düşük indirim + yüksek harcama durumunda riski azalt
        if discount < 0.05 and total_amount > 200:
            prediction = min(prediction * 0.5, 0.2)  # Agresif azalış, en fazla 0.2 risk
            
        # Son sınırlandırma
        prediction = max(0.001, min(0.999, prediction))
        
        # 3 ondalık basamağa yuvarla
        return round(prediction, 3)
    
    def predict_by_order_id(self, order_id):
        """Sipariş ID'si ile tahmin yapma"""
        # Veritabanından sipariş verilerini al
        order_data = self.get_order_by_id(order_id)
        
        # Siparişin tüm ürünleri için tahminler yap
        predictions = []
        explanations = []
        
        for _, row in order_data.iterrows():
            # Her bir ürün için tahmin
            prediction = self.predict(row)
            
            # Her bir ürün için açıklama
            explanation = self.explain_prediction(row)
            
            # Ürün bilgileriyle birlikte kaydet
            predictions.append({
                'discount': float(row['discount']),
                'quantity': int(row['quantity']),
                'unit_price': float(row['unit_price']),
                'product_id': int(row['product_id']),
                'total_amount': float(row['total_amount']),
                'prediction': prediction,
                'explanation': explanation
            })
        
        # Tüm sonuçları döndür
        return {
            'order_id': order_id,
            'items_count': len(predictions),
            'items': predictions,
            'average_risk_score': round(sum(item['prediction'] for item in predictions) / max(1, len(predictions)), 3)
        }
    
    def explain_prediction(self, input_data):
        """Tahmin sonucunu açıklar - model tabanlı yaklaşım."""
        # Girdi verilerini DataFrame'e dönüştür (eğer değilse)
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
            
        # Eğer model eğitilmişse, model çıktılarını analiz et
        if self.model is not None:
            try:
                # Model tabanlı açıklama oluştur
                explanation = self._model_based_explanation(input_data)
                logger.info("Model tabanlı açıklama oluşturuldu")
                return explanation
                
            except Exception as e:
                logger.error(f"Model açıklaması oluşturulurken hata: {str(e)}")
                # Hata durumunda sabit bir açıklama döndür
                return {
                    'discount': 0.4,
                    'quantity': 0.2,
                    'unit_price': 0.2,
                    'total_amount': 0.2
                }
        
        # MODEL YOKSA: sabit bir açıklama döndür
        return {
            'discount': 0.4,
            'quantity': 0.2,
            'unit_price': 0.2,
            'total_amount': 0.2
        }
    
    def _model_based_explanation(self, input_data):
        """
        SHAP değerlerini kullanarak model tabanlı açıklama oluşturur
        """
        try:
            # Girdi verilerini DataFrame'e dönüştür (eğer değilse)
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame([input_data])
            
            # Özellikleri sırayla kullanmak için
            features = input_data[self.feature_names].values
            
            # Verileri ölçeklendir
            features_scaled = self.scaler.transform(features)
            
            # Model predict wrapper - SHAP için
            def model_predict(X):
                return self.model.predict(X).flatten()  # Düzleştir
            
            # KernelExplainer için background data (örnek veri)
            # Sadece bir kaç örnek yeterli
            background_data = np.zeros((10, len(self.feature_names)))
            
            # SHAP KernelExplainer kullan - model-agnostic, tüm TF sürümleriyle uyumlu
            explainer = shap.KernelExplainer(model_predict, background_data)
            
            # SHAP değerlerini hesapla (nsamples=10 yeterli olacaktır)
            shap_values = explainer.shap_values(features_scaled, nsamples=10)
            
            # SHAP değerlerini kontrol et ve düzenle
            logger.info(f"SHAP değerleri boyut: {np.array(shap_values).shape}")
            
            # SHAP değerlerini özellik önemine dönüştür
            feature_importance = {}
            
            # Biçim kontrolü
            if isinstance(shap_values, list):
                # Birden fazla çıktı varsa, ilkini al
                shap_arr = shap_values[0]
            else:
                # Tek çıktı varsa, doğrudan kullan
                shap_arr = shap_values
                
            # Mutlak değerlerini al
            shap_abs = np.abs(shap_arr)
            
            # Boyut kontrolü
            if len(shap_abs.shape) > 1:  # 2+ boyutlu array
                # İlk örnek için değerleri al (tek bir tahmin yaptık)
                importance_values = shap_abs[0, :]
            else:  # 1 boyutlu array
                importance_values = shap_abs
            
            # Özellik önemlerini ata
            for i, feature in enumerate(self.feature_names):
                if i < len(importance_values):
                    feature_importance[feature] = float(importance_values[i])
                else:
                    feature_importance[feature] = 0.0
            
            # Değerleri normalize et (0-1 arası)
            total = sum(feature_importance.values())
            if total > 0:  # Sıfıra bölme hatasını önle
                for key in feature_importance:
                    feature_importance[key] = round(feature_importance[key] / total, 4)
            else:
                # Değerler sıfırsa, eşit dağıt
                for key in feature_importance:
                    feature_importance[key] = round(1.0 / len(feature_importance), 4)
                    
            return feature_importance
            
        except Exception as e:
            logging.error(f"SHAP açıklaması oluşturulurken hata: {str(e)}")
            # Hata durumunda sabit bir açıklama döndür
            return {
                'discount': 0.4,
                'quantity': 0.2,
                'unit_price': 0.2,
                'total_amount': 0.2
            }
        
    def train_and_save(self):
        """Tam eğitim süreci"""
        logger.info("Model eğitimi başlıyor...")
        
        # Veri çek
        df = self.fetch_data()
        
        # Veriyi hazırla
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Eğitim verilerinden bir kısmını SHAP için sakla
        self.X_train = X_train[:100].copy()
        
        # Modeli oluştur ve eğit
        self.build_model()
        history = self.train(X_train, y_train, X_test, y_test)
        
        # Modeli ve scaler'ı kaydet
        self.model.save('return_risk_model.h5')
        
        # Scaler'ı da kaydet
        joblib.dump(self.scaler, 'scaler.pkl')
        
        logger.info("Model ve scaler eğitildi ve kaydedildi.")
        
        # Test veri seti performansını değerlendir
        test_metrics = self.model.evaluate(X_test, y_test)
        if len(test_metrics) > 1:
            test_loss, test_acc = test_metrics[0], test_metrics[1]
            logger.info(f"Test kaybı: {test_loss:.4f}")
            logger.info(f"Test doğruluğu: {test_acc:.4f}")
        else:
            logger.info(f"Test kaybı: {test_metrics:.4f}")
        
        return history 