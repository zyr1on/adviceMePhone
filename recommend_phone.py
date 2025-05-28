#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import sys
import os
import pandas as pd
from datetime import datetime
import re

class PhonePredictor:
    def __init__(self, model_path="enhanced_phone_model.pkl", data_path="phones.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.phones_df = None
        self.load_model()
        self.load_data()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ HATA: Model dosyası '{self.model_path}' bulunamadı!")
                print("Önce modeli eğitmeniz gerekiyor.")
                sys.exit(1)
            
            print("🔄 Model yükleniyor...")
            model_data = joblib.load(self.model_path)
            
            self.vectorizer = model_data['vectorizer']
            self.models = model_data['models']
            self.label_encoders = model_data['label_encoders']
            self.feature_selectors = model_data.get('feature_selectors', {})
            self.feature_names = model_data['feature_names']
            self.best_params = model_data.get('best_params', {})
            self.pattern_dict = model_data.get('pattern_dict', {})
            
            print("✅ Model başarıyla yüklendi!")
            print(f"📊 Desteklenen özellikler: {', '.join(self.feature_names)}")
            
        except Exception as e:
            print(f"❌ Model yüklenirken hata: {e}")
            sys.exit(1)
    
    def load_data(self):
        """Veri setini yükle ve temizle"""
        try:
            print("🔄 Veri seti yükleniyor...")
            self.phones_df = pd.read_csv(self.data_path)
            
            # Sütun adlarını küçük harfe çevir
            self.phones_df.columns = [col.lower() for col in self.phones_df.columns]
            
            # noneP ve eksik değerleri temizle
            self.phones_df = self.phones_df.replace('noneP', 'none')
            self.phones_df = self.phones_df.fillna('none')
            
            # price ve ram sütunlarını float'a çevir
            self.phones_df['price'] = pd.to_numeric(self.phones_df['price'], errors='coerce').fillna(0.0)
            self.phones_df['ram'] = pd.to_numeric(self.phones_df['ram'], errors='coerce').fillna(0.0)
            
            print("✅ Veri seti başarıyla yüklendi!")
            print(f"📱 Toplam {len(self.phones_df)} telefon yüklendi.")
            
        except Exception as e:
            print(f"❌ Veri seti yüklenirken hata: {e}")
            sys.exit(1)
    
    def advanced_preprocess_text(self, text):
        """Gelişmiş metin ön işleme"""
        text = text.lower()
        
        # Türkçe karakter dönüşümü
        turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
        for tr_char, en_char in turkish_chars.items():
            text = text.replace(tr_char, en_char)
        
        # Yazım hatalarını düzelt
        text = text.replace('andorid', 'android')  # 'andorid' -> 'android'
        text = text.replace('kamerasi', 'kamera')  # 'kamerası' -> 'kamera'
        
        # Sayıları normalize et
        text = re.sub(r'\b(\d+)\s*(gb|tb|mp|mah|tl|lira)\b', r'\1\2', text)
        
        # Özel kalıpları işaretle
        for feature, pattern in self.pattern_dict.items():
            if re.search(pattern, text, re.IGNORECASE):
                text += f' FEATURE_{feature.upper()}'
        
        return text
    
    def predict(self, input_text):
        """Tahmin yap"""
        processed_text = self.advanced_preprocess_text(input_text)
        X_vectorized = self.vectorizer.transform([processed_text])
        
        predictions = {}
        confidences = {}
        
        for feature in self.feature_names:
            if feature not in self.models:
                predictions[feature] = 'none'
                confidences[feature] = 0.0
                continue
            
            try:
                if self.feature_selectors.get(feature):
                    X_selected = self.feature_selectors[feature].transform(X_vectorized)
                else:
                    X_selected = X_vectorized
                
                pred_encoded = self.models[feature].predict(X_selected)[0]
                pred_proba = self.models[feature].predict_proba(X_selected)[0]
                
                prediction = self.label_encoders[feature].inverse_transform([pred_encoded])[0]
                # usage için photo -> camera düzeltmesi
                if feature == 'usage' and prediction == 'photo':
                    prediction = 'camera'
                predictions[feature] = prediction
                confidences[feature] = max(pred_proba)
                
            except Exception as e:
                print(f"⚠️  '{feature}' için tahmin hatası: {e}")
                predictions[feature] = 'none'
                confidences[feature] = 0.0
        
        return predictions, confidences
    
    def recommend_phones(self, predictions, confidences, top_n=None):
        """Tahminlere göre telefon öner"""
        filtered_df = self.phones_df.copy()
        
        # Tahminlere göre filtreleme
        for feature, value in predictions.items():
            if value != 'none' and feature in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[feature] == value]
        
        # Filtrelenmiş telefonları fiyat sırasına göre sırala
        try:
            filtered_df = filtered_df.sort_values(by='price')
        except Exception as e:
            print(f"⚠️ Sıralama hatası: {e}")
            print("Veri setindeki 'price' sütununda veri tipi uyumsuzluğu olabilir.")
            # Hata durumunda, fiyatları float'a çevirmeye çalış
            filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce').fillna(0.0)
            filtered_df = filtered_df.sort_values(by='price')
        
        # top_n None ise tüm eşleşmeleri göster, yoksa ilk top_n kadar
        recommendations = filtered_df if top_n is None else filtered_df.head(top_n)
        
        # Önerileri formatla
        print("\n" + "="*60)
        print("📱 ÖNERİLEN TELEFONLAR")
        print("="*60)
        
        if recommendations.empty:
            print("😕 Üzgünüz, kriterlerinize uygun telefon bulunamadı.")
            print("Daha farklı özellikler deneyebilirsiniz.")
        else:
            for idx, row in recommendations.iterrows():
                phone_info = (
                    f"{row['brand']} - {row['price']} TL\n"
                    f"📱 İşletim Sistemi: {row['os']}\n"
                    f"💾 Depolama: {row['storage']}\n"
                    f"🧠 RAM: {row['ram']}GB\n"
                    f"📸 Kamera: {row['camera']}\n"
                    f"🔋 Batarya: {row['battery']}\n"
                    f"📺 Ekran: {row['screen']}\n"
                    f"🎯 Kullanım Amacı: {row['usage']}"
                )
                print(f"{idx + 1}. {phone_info}")
                print("-"*60)
        
        # Önerilen telefon sayısını göster
        print(f"📋 Toplam {len(recommendations)} telefon önerildi.")
        
        # Ortalama güven skoru
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0
        print(f"📊 Ortalama Güven Skoru: {avg_confidence:.2f}")
        if avg_confidence >= 0.7:
            print("✅ Yüksek güvenilirlik - Öneriler oldukça kesin!")
        elif avg_confidence >= 0.5:
            print("⚠️ Orta güvenilirlik - Öneriler makul seviyede.")
        else:
            print("❌ Düşük güvenilirlik - Daha spesifik bilgi vermeyi deneyin.")
        
        return recommendations
    
    def format_output(self, predictions, confidences, recommendations):
        """Sonuçları güzel formatta göster"""
        print("\n" + "="*60)
        print("🤖 MODEL HAM ÇIKTISI")
        print("="*60)
        
        raw_output_parts = []
        for feature in self.feature_names:
            pred = predictions.get(feature, 'none')
            raw_output_parts.append(f"{feature}:{pred}")
        
        raw_output = "; ".join(raw_output_parts)
        print(f"📤 {raw_output}")
        
        print("\n" + "="*60)
        print("📱 TELEFON ÖNERİ SONUÇLARI (DETAYLI)")
        print("="*60)
        
        turkish_names = {
            'price': '💰 Fiyat Aralığı',
            'brand': '🏢 Marka Tercihi',
            'os': '📱 İşletim Sistemi',
            'usage': '🎯 Kullanım Amacı',
            'ram': '🧠 RAM Kapasitesi',
            'storage': '💾 Depolama Alanı',
            'battery': '🔋 Batarya Gücü',
            'camera': '📸 Kamera Kalitesi',
            'screen': '📺 Ekran Boyutu'
        }
        
        sorted_features = sorted(
            self.feature_names, 
            key=lambda x: confidences.get(x, 0), 
            reverse=True
        )
        
        for feature in sorted_features:
            name = turkish_names.get(feature, feature.title())
            pred = predictions.get(feature, 'none')
            conf = confidences.get(feature, 0.0)
            
            confidence_icon = "🟢 Yüksek" if conf >= 0.8 else "🟡 Orta" if conf >= 0.6 else "🟠 Düşük" if conf >= 0.4 else "🔴 Çok Düşük"
            print(f"{name:<20}: {pred:<15} | Güven: {conf:.2f} {confidence_icon}")
        
        print("="*60)
        
        # Önerileri göster
        self.recommend_phones(predictions, confidences)
    
    def interactive_mode(self):
        """Etkileşimli mod"""
        print("\n🎯 TELEFON ÖNERİ SİSTEMİ")
        print("="*50)
        print("İstediğiniz telefon özelliklerini yazın...")
        print("Örnek: 'ucuz gaming telefon 8gb ram büyük ekran'")
        print("Çıkmak için 'q' yazın.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n💬 Telefon tercihinizi yazın: ").strip()
                
                if user_input.lower() in ['q', 'quit', 'exit', 'çık', 'cik']:
                    print("👋 İyi günler!")
                    break
                
                if not user_input:
                    print("⚠️ Lütfen bir şeyler yazın...")
                    continue
                
                print(f"\n🔍 Analiz ediliyor: '{user_input}'")
                
                predictions, confidences = self.predict(user_input)
                self.format_output(predictions, confidences, None)
                
                self.save_prediction_log(user_input, predictions, confidences)
                
            except KeyboardInterrupt:
                print("\n\n👋 Programa son verildi. İyi günler!")
                break
            except Exception as e:
                print(f"❌ Hata oluştu: {e}")
    
    def save_prediction_log(self, input_text, predictions, confidences):
        """Tahminleri log dosyasına kaydet"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open("prediction_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}]\n")
                f.write(f"Girdi: {input_text}\n")
                f.write("Tahminler:\n")
                
                for feature in self.feature_names:
                    pred = predictions.get(feature, 'none')
                    conf = confidences.get(feature, 0.0)
                    f.write(f"  {feature}: {pred} ({conf:.2f})\n")
                f.write("-" * 50 + "\n")
                
        except Exception as e:
            print(f"⚠️ Log kaydedilirken hata: {e}")

def main():
    """Ana fonksiyon"""
    print("🚀 Telefon Öneri Sistemi Başlatılıyor...")
    
    model_path = "enhanced_phone_model.pkl"
    data_path = "phones.csv"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    
    try:
        predictor = PhonePredictor(model_path, data_path)
        
        if len(sys.argv) > 3:
            input_text = " ".join(sys.argv[3:])
            print(f"🔍 Tek tahmin modu: '{input_text}'")
            
            predictions, confidences = predictor.predict(input_text)
            predictor.format_output(predictions, confidences, None)
        else:
            predictor.interactive_mode()
            
    except Exception as e:
        print(f"❌ Program hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
