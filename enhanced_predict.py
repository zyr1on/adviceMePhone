#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import sys
import os
from datetime import datetime

class PhonePredictor:
    def __init__(self, model_path="enhanced_phone_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ HATA: Model dosyası '{self.model_path}' bulunamadı!")
                print("Önce modeli eğitmeniz gerekiyor.")
                sys.exit(1)
            
            print("🔄 Model yükleniyor...")
            model_data = joblib.load(self.model_path)
            
            # Model bileşenlerini yükle
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
    
    def advanced_preprocess_text(self, text):
        """Gelişmiş metin ön işleme"""
        import re
        
        text = text.lower()
        
        # Türkçe karakter dönüşümü
        turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
        for tr_char, en_char in turkish_chars.items():
            text = text.replace(tr_char, en_char)
        
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
                predictions[feature] = 'belirsiz'
                confidences[feature] = 0.0
                continue
            
            try:
                # Özellik seçimi uygula
                if self.feature_selectors.get(feature):
                    X_selected = self.feature_selectors[feature].transform(X_vectorized)
                else:
                    X_selected = X_vectorized
                
                # Tahmin
                pred_encoded = self.models[feature].predict(X_selected)[0]
                pred_proba = self.models[feature].predict_proba(X_selected)[0]
                
                prediction = self.label_encoders[feature].inverse_transform([pred_encoded])[0]
                confidence = max(pred_proba)
                
                predictions[feature] = prediction
                confidences[feature] = confidence
                
            except Exception as e:
                print(f"⚠️  '{feature}' için tahmin hatası: {e}")
                predictions[feature] = 'hata'
                confidences[feature] = 0.0
        
        return predictions, confidences
    
    def format_output(self, predictions, confidences):
        """Sonuçları güzel formatta göster"""
        
        # Önce modelin ham çıktısını göster
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
        
        # Özellik isimleri Türkçe karşılıkları
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
        
        # Güven skoruna göre sırala
        sorted_features = sorted(
            self.feature_names, 
            key=lambda x: confidences.get(x, 0), 
            reverse=True
        )
        
        for feature in sorted_features:
            name = turkish_names.get(feature, feature.title())
            pred = predictions.get(feature, 'belirsiz')
            conf = confidences.get(feature, 0.0)
            
            # Güven seviyesi göstergesi
            if conf >= 0.8:
                confidence_icon = "🟢 Yüksek"
            elif conf >= 0.6:
                confidence_icon = "🟡 Orta"
            elif conf >= 0.4:
                confidence_icon = "🟠 Düşük"
            else:
                confidence_icon = "🔴 Çok Düşük"
            
            print(f"{name:<20}: {pred:<15} | Güven: {conf:.2f} {confidence_icon}")
        
        print("="*60)
        
        # Ham çıktıyı tekrar göster (kolay kopyalama için)
        print(f"\n📋 MODEL ÇIKTISI (Kopyalanabilir):")
        raw_output_parts = []
        for feature in self.feature_names:
            pred = predictions.get(feature, 'none')
            raw_output_parts.append(f"{feature}:{pred}")
        print(f"🔗 {'; '.join(raw_output_parts)}")
        
        # Genel değerlendirme
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0
        print(f"\n📊 Ortalama Güven Skoru: {avg_confidence:.2f}")
        
        if avg_confidence >= 0.7:
            print("✅ Yüksek güvenilirlik - Tahminler oldukça kesin!")
        elif avg_confidence >= 0.5:
            print("⚠️  Orta güvenilirlik - Tahminler makul seviyede.")
        else:
            print("❌ Düşük güvenilirlik - Daha spesifik bilgi vermeyi deneyin.")
    
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
                    print("⚠️  Lütfen bir şeyler yazın...")
                    continue
                
                print(f"\n🔍 Analiz ediliyor: '{user_input}'")
                
                # Tahmin yap
                predictions, confidences = self.predict(user_input)
                
                # Sonuçları göster
                self.format_output(predictions, confidences)
                
                # Önerileri kaydet
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
                    pred = predictions.get(feature, 'belirsiz')
                    conf = confidences.get(feature, 0.0)
                    f.write(f"  {feature}: {pred} ({conf:.2f})\n")
                f.write("-" * 50 + "\n")
                
        except Exception as e:
            print(f"⚠️  Log kaydedilirken hata: {e}")

def main():
    """Ana fonksiyon"""
    print("🚀 Telefon Öneri Sistemi Başlatılıyor...")
    
    # Model dosya yolunu kontrol et
    model_path = "enhanced_phone_model.pkl"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        # Predictor'ı başlat
        predictor = PhonePredictor(model_path)
        
        # Eğer komut satırından argüman verilmişse tek tahmin yap
        if len(sys.argv) > 2:
            input_text = " ".join(sys.argv[2:])
            print(f"🔍 Tek tahmin modu: '{input_text}'")
            
            predictions, confidences = predictor.predict(input_text)
            predictor.format_output(predictions, confidences)
        else:
            # Etkileşimli mod
            predictor.interactive_mode()
            
    except Exception as e:
        print(f"❌ Program hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

