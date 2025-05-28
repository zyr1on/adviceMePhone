import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class PhoneRecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.models = {}
        self.label_encoders = {}
        self.feature_names = ['price', 'brand', 'os', 'usage', 'ram', 'storage', 'battery', 'camera', 'screen']
        
    def preprocess_text(self, text):
        # Türkçe karakterleri normalize et
        text = text.lower()
        text = text.replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
        text = text.replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
        return text
    
    def extract_features_from_output(self, output_text):
        features = {f: 'none' for f in self.feature_names}
        parts = output_text.split(';')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in self.feature_names:
                    features[key] = value
        return features
    
    def prepare_training_data(self, data_file):
        X_texts = []
        y_dict = {f: [] for f in self.feature_names}
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if '->' not in line:
                continue
            input_text, output_text = map(str.strip, line.split('->', 1))
            processed_input = self.preprocess_text(input_text)
            X_texts.append(processed_input)
            
            features = self.extract_features_from_output(output_text)
            for f in self.feature_names:
                y_dict[f].append(features[f])
        
        return X_texts, y_dict
    
    def train(self, data_file):
        print("Veri hazırlanıyor...")
        X, y_dict = self.prepare_training_data(data_file)
        
        print(f"{len(X)} adet örnek bulundu.")
        print("Metinler vektörleştiriliyor...")
        X_vectorized = self.vectorizer.fit_transform(X)
        
        for feature in self.feature_names:
            print(f"'{feature}' özelliği için model eğitiliyor...")
            le = LabelEncoder()
            y = le.fit_transform(y_dict[feature])
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_vectorized, y)
            self.models[feature] = model
            self.label_encoders[feature] = le
        
        print("Tüm modeller eğitildi.")
    
    def save_model(self, path):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        print(f"Model '{path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    model = PhoneRecommendationModel()
    model.train("training_data.txt")
    model.save_model("phone_model.pkl")
