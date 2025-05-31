try:
    import joblib
    import os
    import pandas as pd
    import re
    
except ImportError as e:
    print("\n[!] Gerekli modüller yüklenemedi:")
    print(f"    -> {e}")
    print("\nLütfen gerekli bağımlılıkları yüklemek için aşağıdaki komutu çalıştırın:")
    print("    python3 install.py\n")


class PhonePredictor:
    def __init__(self, model_path="enhanced_phone_model.pkl", data_path="phones.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.phones_df = None
        self.load_model()
        self.load_data()

    def load_model(self):
        """Modeli yükler ve gerekli bileşenleri atar"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model dosyası '{self.model_path}' bulunamadı!")
            
            model_data = joblib.load(self.model_path)
            self.vectorizer = model_data['vectorizer']
            self.models = model_data['models']
            self.label_encoders = model_data['label_encoders']
            self.feature_selectors = model_data.get('feature_selectors', {})
            self.feature_names = model_data['feature_names']
            self.best_params = model_data.get('best_params', {})
            self.pattern_dict = model_data.get('pattern_dict', {})

        except Exception as e:
            raise Exception(f"Model yüklenirken hata: {e}")

    def load_data(self):
        """Telefon verilerini yükler ve gerekli ön işlemleri yapar"""
        try:
            self.phones_df = pd.read_csv(self.data_path)
            self.phones_df.columns = [col.lower().strip() for col in self.phones_df.columns]
            if 'camera' in self.phones_df.columns:
                self.phones_df.rename(columns={'camera': 'camera'}, inplace=True)
            self.phones_df = self.phones_df.replace('noneP', 'none')
            self.phones_df = self.phones_df.fillna('none')
            self.phones_df['price'] = pd.to_numeric(self.phones_df['price'], errors='coerce').fillna(0.0)
            self.phones_df['ram'] = pd.to_numeric(self.phones_df['ram'], errors='coerce').fillna(0.0)
            for col in ['os', 'storage', 'camera', 'battery', 'screen', 'usage']:
                if col in self.phones_df.columns:
                    self.phones_df[col] = self.phones_df[col].str.lower().str.strip()
        except Exception as e:
            raise Exception(f"Veri seti yüklenirken hata: {e}")

    def advanced_preprocess_text(self, text):
        """Kullanıcıdan alınan metni işleyip, çıkartılan özellikleri döndürür"""
        text = text.lower().strip()
        
        turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
        for tr_char, en_char in turkish_chars.items():
            text = text.replace(tr_char, en_char)
        
        text = text.replace('andorid', 'android')
        text = text.replace('kamerasi', 'kamera')
        
        ram_match = re.search(r'(\d+)\s*gb\s*(ram)?', text, re.IGNORECASE)
        price_match = re.search(r'(\d+)\s*(tl|lira)', text, re.IGNORECASE)
        
        extracted_values = {}
        if ram_match:
            extracted_values['ram'] = float(ram_match.group(1))
            text = text.replace(ram_match.group(0), 'ram')
        if price_match:
            extracted_values['price'] = float(price_match.group(1))
            text = text.replace(price_match.group(0), 'price')
        
        text = re.sub(r'\b(\d+)\s*(gb|tb|mp|mah|tl|lira)\b', r'\1\2', text)
        
        for feature, pattern in self.pattern_dict.items():
            if re.search(pattern, text, re.IGNORECASE):
                text += f' FEATURE_{feature.upper()}'
        
        return text, extracted_values

    def predict(self, input_text):
        """Verilen metni işleyip, tahminleri döndürür"""
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
                    if feature == 'usage' and prediction == 'photo':
                        prediction = 'camera'
                    predictions[feature] = prediction
                    confidences[feature] = max(pred_proba)
                    
                except Exception as e:
                    predictions[feature] = 'none'
                    confidences[feature] = 0.0
            
            predictions_list.append(predictions)
            confidences_list.append(confidences)
            raw_outputs.append(f"{feature_text}: " + "; ".join(f"{k}:{v}" for k, v in predictions.items()))
            extracted_values_list.append(extracted_values)
        
        return predictions_list, confidences_list, raw_outputs, extracted_values_list

    def recommend_phones(self, predictions_list, confidences_list, extracted_values_list, top_n=None):
        """Önerilen telefonları döndürür"""
        filtered_df = self.phones_df.copy()
        applied_filters = []
        
        for idx, (predictions, extracted_values) in enumerate(zip(predictions_list, extracted_values_list)):
            for feature, value in predictions.items():
                if value != 'none' and feature in filtered_df.columns and feature not in ['price', 'ram']:
                    try:
                        filtered_df = filtered_df[filtered_df[feature] == value]
                        applied_filters.append(f"{feature}={value}")
                    except:
                        continue
            
            if 'ram' in extracted_values:
                ram_value = extracted_values['ram']
                try:
                    filtered_df = filtered_df[filtered_df['ram'] == ram_value]
                    applied_filters.append(f"ram={ram_value}GB")
                except:
                    pass
            
            if 'price' in extracted_values:
                price_value = extracted_values['price']
                try:
                    filtered_df = filtered_df[filtered_df['price'] <= price_value]
                    applied_filters.append(f"price<={price_value}TL")
                except:
                    pass
        
        try:
            filtered_df = filtered_df.sort_values(by='price')
        except:
            filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce').fillna(0.0)
            filtered_df = filtered_df.sort_values(by='price')
        
        recommendations = filtered_df if top_n is None else filtered_df.head(top_n)
        
        return recommendations
