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
            
            # Sütun adlarını küçük harfe çevir ve boşlukları kaldır
            self.phones_df.columns = [col.lower().strip() for col in self.phones_df.columns]
            
            # Camera sütununu camera olarak yeniden adlandır
            if 'camera' in self.phones_df.columns:
                self.phones_df.rename(columns={'camera': 'camera'}, inplace=True)
            
            # noneP ve eksik değerleri temizle
            self.phones_df = self.phones_df.replace('noneP', 'none')
            self.phones_df = self.phones_df.fillna('none')
            
            # price ve ram sütunlarını float'a çevir
            self.phones_df['price'] = pd.to_numeric(self.phones_df['price'], errors='coerce').fillna(0.0)
            self.phones_df['ram'] = pd.to_numeric(self.phones_df['ram'], errors='coerce').fillna(0.0)
            
            # Veri setindeki değerleri normalize et
            for col in ['os', 'storage', 'camera', 'battery', 'screen', 'usage']:
                if col in self.phones_df.columns:
                    self.phones_df[col] = self.phones_df[col].str.lower().str.strip()
            
            print("✅ Veri seti başarıyla yüklendi!")
            print(f"📱 Toplam {len(self.phones_df)} telefon yüklendi.")
            print(f"📋 Sütunlar: {', '.join(self.phones_df.columns)}")
            
        except Exception as e:
            print(f"❌ Veri seti yüklenirken hata: {e}")
            sys.exit(1)
    
    def advanced_preprocess_text(self, text):
        """Gelişmiş metin ön işleme ve sayısal değer çıkarma"""
        text = text.lower().strip()
        
        # Türkçe karakter dönüşümü
        turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
        for tr_char, en_char in turkish_chars.items():
            text = text.replace(tr_char, en_char)
        
        # Yazım hatalarını düzelt
        text = text.replace('andorid', 'android')
        text = text.replace('kamerasi', 'kamera')
        
        # Sayısal değerleri çıkar
        ram_match = re.search(r'(\d+)\s*gb\s*(ram)?', text, re.IGNORECASE)
        price_match = re.search(r'(\d+)\s*(tl|lira)', text, re.IGNORECASE)
        
        extracted_values = {}
        if ram_match:
            extracted_values['ram'] = float(ram_match.group(1))
            text = text.replace(ram_match.group(0), 'ram')
        if price_match:
            extracted_values['price'] = float(price_match.group(1))
            text = text.replace(price_match.group(0), 'price')
        
        # Sayıları normalize et
        text = re.sub(r'\b(\d+)\s*(gb|tb|mp|mah|tl|lira)\b', r'\1\2', text)
        
        # Özel kalıpları işaretle
        for feature, pattern in self.pattern_dict.items():
            if re.search(pattern, text, re.IGNORECASE):
                text += f' FEATURE_{feature.upper()}'
        
        return text, extracted_values
    
    def predict(self, input_text):
        """Tahmin yap"""
        # Girdiyi nokta veya virgülle ayır
        features = [feat.strip() for feat in re.split(r'[,.]', input_text) if feat.strip()]
        
        if not features:
            return {}, {}, [], {}
        
        predictions_list = []
        confidences_list = []
        raw_outputs = []
        extracted_values_list = []
        
        for feature_text in features:
            processed_text, extracted_values = self.advanced_preprocess_text(feature_text)
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
            
            predictions_list.append(predictions)
            confidences_list.append(confidences)
            raw_outputs.append(f"📤 {feature_text}: " + "; ".join(f"{k}:{v}" for k, v in predictions.items()))
            extracted_values_list.append(extracted_values)
        
        return predictions_list, confidences_list, raw_outputs, extracted_values_list
    
    def recommend_phones(self, predictions_list, confidences_list, extracted_values_list, top_n=None):
        """Tahminlere ve sayısal değerlere göre telefon öner"""
        filtered_df = self.phones_df.copy()
        applied_filters = []
        
        # Her bir özelliğin tahminlerini ve sayısal değerlerini sırayla filtrele
        for idx, (predictions, confidences, extracted_values) in enumerate(zip(predictions_list, confidences_list, extracted_values_list)):
            print(f"\n🔍 Özellik {idx + 1} için filtreleme yapılıyor...")
            initial_count = len(filtered_df)
            
            # Kategorik özellikler için filtreleme
            for feature, value in predictions.items():
                if value != 'none' and feature in filtered_df.columns and feature not in ['price', 'ram']:
                    try:
                        filtered_df = filtered_df[filtered_df[feature] == value]
                        applied_filters.append(f"{feature}={value}")
                    except Exception as e:
                        print(f"⚠️ '{feature}' için filtreleme hatası: {e}")
                        continue
            
            # Sayısal özellikler için filtreleme
            if 'ram' in extracted_values:
                ram_value = extracted_values['ram']
                try:
                    filtered_df = filtered_df[filtered_df['ram'] == ram_value]
                    applied_filters.append(f"ram={ram_value}GB")
                except Exception as e:
                    print(f"⚠️ 'ram' için filtreleme hatası: {e}")
            
            if 'price' in extracted_values:
                price_value = extracted_values['price']
                try:
                    filtered_df = filtered_df[filtered_df['price'] <= price_value]
                    applied_filters.append(f"price<={price_value}TL")
                except Exception as e:
                    print(f"⚠️ 'price' için filtreleme hatası: {e}")
            
            final_count = len(filtered_df)
            print(f"📊 {initial_count} telefondan {final_count} telefona düşüldü.")
            if final_count == 0:
                print(f"😕 Özellik {idx + 1} için uygun telefon bulunamadı.")
                print(f"🛠️ Uygulanan filtreler: {', '.join(applied_filters) if applied_filters else 'Yok'}")
                # Alternatif öneriler
                if 'ram' in extracted_values and filtered_df.empty:
                    closest_rams = self.phones_df['ram'].unique()
                    closest_rams = sorted([x for x in closest_rams if isinstance(x, float)], key=lambda x: abs(x - ram_value))
                    if closest_rams:
                        print(f"📢 '{ram_value}GB RAM' bulunamadı. En yakın RAM değerleri: {', '.join([str(x) + 'GB' for x in closest_rams[:3]])}")
                if 'price' in extracted_values and filtered_df.empty:
                    closest_prices = self.phones_df[self.phones_df['price'] > price_value]['price'].unique()
                    closest_prices = sorted(closest_prices)[:3]
                    if closest_prices:
                        print(f"📢 '{price_value}TL' altında telefon bulunamadı. En yakın fiyatlar: {', '.join([str(x) + 'TL' for x in closest_prices])}")
                return pd.DataFrame()
        
        # Filtrelenmiş telefonları fiyat sırasına göre sırala
        try:
            filtered_df = filtered_df.sort_values(by='price')
        except Exception as e:
            print(f"⚠️ Sıralama hatası: {e}")
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
        print(f"🛠️ Uygulanan filtreler: {', '.join(applied_filters) if applied_filters else 'Yok'}")
        
        # Ortalama güven skoru
        avg_confidence = sum(sum(c.values()) / len(c) for c in confidences_list) / len(confidences_list) if confidences_list else 0
        print(f"📊 Ortalama Güven Skoru: {avg_confidence:.2f}")
        if avg_confidence >= 0.7:
            print("✅ Yüksek güvenilirlik - Öneriler oldukça kesin!")
        elif avg_confidence >= 0.5:
            print("⚠️ Orta güvenilirlik - Öneriler makul seviyede.")
        else:
            print("❌ Düşük güvenilirlik - Daha spesifik bilgi vermeyi deneyin.")
        
        return recommendations
    
    def format_output(self, predictions_list, confidences_list, raw_outputs, extracted_values_list, recommendations):
        """Sonuçları güzel formatta göster"""
        for idx, (raw_output, predictions, confidences, extracted_values) in enumerate(zip(raw_outputs, predictions_list, confidences_list, extracted_values_list)):
            print("\n" + "="*60)
            print(f"🤖 ÖZELLİK {idx + 1} HAM ÇIKTISI")
            print("="*60)
            print(raw_output)
            if extracted_values:
                print(f"📊 Çıkarılan Sayısal Değerler: {', '.join(f'{k}={v}' for k, v in extracted_values.items())}")
            
            print("\n" + "="*60)
            print(f"📱 ÖZELLİK {idx + 1} TAHMİN SONUÇLARI (DETAYLI)")
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
        self.recommend_phones(predictions_list, confidences_list, extracted_values_list)
    
    def interactive_mode(self):
        """Etkileşimli mod"""
        print("\n🎯 TELEFON ÖNERİ SİSTEMİ")
        print("="*50)
        print("İstediğiniz telefon özelliklerini yazın, birden fazla özellik için virgül veya nokta kullanın...")
        print("Örnek: 'kamerası iyi olsun, ios olsun, 8gb ram, 15000 tl'")
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
                
                predictions_list, confidences_list, raw_outputs, extracted_values_list = self.predict(user_input)
                self.format_output(predictions_list, confidences_list, raw_outputs, extracted_values_list, None)
                
                self.save_prediction_log(user_input, predictions_list, confidences_list, extracted_values_list)
                
            except KeyboardInterrupt:
                print("\n\n👋 Programa son verildi. İyi günler!")
                break
            except Exception as e:
                print(f"❌ Hata oluştu: {e}")
    
    def save_prediction_log(self, input_text, predictions_list, confidences_list, extracted_values_list):
        """Tahminleri log dosyasına kaydet"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open("prediction_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}]\n")
                f.write(f"Girdi: {input_text}\n")
                for idx, (predictions, confidences, extracted_values) in enumerate(zip(predictions_list, confidences_list, extracted_values_list)):
                    f.write(f"Özellik {idx + 1} Tahminleri:\n")
                    for feature in self.feature_names:
                        pred = predictions.get(feature, 'none')
                        conf = confidences.get(feature, 0.0)
                        f.write(f"  {feature}: {pred} ({conf:.2f})\n")
                    if extracted_values:
                        f.write(f"  Çıkarılan Sayısal Değerler: {', '.join(f'{k}={v}' for k, v in extracted_values.items())}\n")
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
            
            predictions_list, confidences_list, raw_outputs, extracted_values_list = predictor.predict(input_text)
            predictor.format_output(predictions_list, confidences_list, raw_outputs, extracted_values_list, None)
        else:
            predictor.interactive_mode()
            
    except Exception as e:
        print(f"❌ Program hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
