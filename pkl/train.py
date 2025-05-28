import joblib
import numpy as np
import re

class PhonePredictor:
    def __init__(self, model_path):
        print("Model yükleniyor...")
        model_data = joblib.load(model_path)
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print("Model başarıyla yüklendi!")
    
    def preprocess_text(self, text):
        # Türkçe karakterleri normalize et
        text = text.lower()
        text = text.replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
        text = text.replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
        return text
    
    def predict(self, input_text):
        # Preprocess input
        processed_text = self.preprocess_text(input_text)
        
        # Vectorize input
        X_vec = self.vectorizer.transform([processed_text])
        
        # Predict for each feature
        predictions = {}
        confidence_scores = {}
        
        for feature in self.feature_names:
            if feature in self.models:
                # Get prediction
                pred_encoded = self.models[feature].predict(X_vec)[0]
                pred_proba = self.models[feature].predict_proba(X_vec)[0]
                
                # Decode prediction
                try:
                    prediction = self.label_encoders[feature].inverse_transform([pred_encoded])[0]
                    confidence = np.max(pred_proba)
                    
                    # Only include if confidence is reasonable and not 'none'
                    if confidence > 0.1 and prediction != 'none':
                        predictions[feature] = prediction
                        confidence_scores[feature] = confidence
                except:
                    continue
        
        return predictions, confidence_scores
    
    def format_output(self, predictions):
        if not predictions:
            return "Belirtilen kriterler anlaşılamadı."
        
        output_parts = []
        for feature, value in predictions.items():
            output_parts.append(f"{feature}: {value}")
        
        return "; ".join(output_parts)

def main():
    try:
        predictor = PhonePredictor('phone_model.pkl')
        
        print("Telefon Öneri Sistemi")
        print("Çıkmak için 'quit' yazın")
        print("-" * 50)
        
        while True:
            user_input = input("\nTelefon isteğinizi yazın: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("Görüşürüz!")
                break
            
            if not user_input:
                print("Lütfen bir istek yazın.")
                continue
            
            # Predict
            predictions, confidence_scores = predictor.predict(user_input)
            
            # Format and display output
            output = predictor.format_output(predictions)
            print(f"\nSonuç: {output}")
            
            # Show confidence scores if desired
            if predictions:
                print("\nGüven skorları:")
                for feature, confidence in confidence_scores.items():
                    print(f"  {feature}: {confidence:.2f}")
    
    except FileNotFoundError:
        print("Model dosyası bulunamadı! Önce train.py'yi çalıştırın.")
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
