import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AdvancedPhoneRecommendationModel:
    def __init__(self):
        # Daha kapsamlı TF-IDF parametreleri
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Özellik sayısını artırdık
            ngram_range=(1, 4),  # 4-gram'a kadar
            min_df=2,  # En az 2 dokümanda geçen kelimeler
            max_df=0.8,  # Çok yaygın kelimeleri filtrele
            stop_words=self._get_turkish_stopwords(),
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        self.models = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.best_params = {}
        
        self.feature_names = [
            'price', 'brand', 'os', 'usage', 'ram', 
            'storage', 'battery', 'camera', 'screen'
        ]
        
        # Türkçe özel kelime kalıpları
        self.pattern_dict = {
            'price': r'(?:ucuz|pahalı|ekonomik|bütçe|fiyat|tl|lira|\d+\s*bin)',
            'brand': r'(?:samsung|apple|xiaomi|huawei|oppo|iphone|galaxy)',
            'os': r'(?:android|ios|iphone)',
            'usage': r'(?:oyun|fotoğraf|iş|günlük|sosyal|video|müzik)',
            'ram': r'(?:\d+\s*gb\s*ram|bellek|\d+gb)',
            'storage': r'(?:depolama|hafıza|\d+\s*gb|\d+\s*tb)',
            'battery': r'(?:batarya|pil|şarj|mah|\d+\s*mah)',
            'camera': r'(?:kamera|fotoğraf|çekim|mp|\d+\s*mp|selfie)',
            'screen': r'(?:ekran|inch|inç|\d+\.?\d*\s*inch|\d+\.?\d*\s*inç)'
        }
    
    def _get_turkish_stopwords(self):
        return [
            've', 'bir', 'bu', 'da', 'de', 'en', 'ile', 'için', 'mi', 'mu', 'mü',
            'ne', 'olan', 'olarak', 'var', 'ya', 'daha', 'çok', 'iyi', 'güzel',
            'şey', 'şu', 'o', 'ben', 'sen', 've', 'ama', 'ancak', 'çünkü'
        ]
    
    def advanced_preprocess_text(self, text):
        # Gelişmiş metin ön işleme
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
    
    def extract_enhanced_features(self, output_text):
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
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Hata: '{data_file}' dosyası bulunamadı!")
            return [], {}
        
        valid_samples = 0
        for line in lines:
            line = line.strip()
            if not line or '->' not in line:
                continue
                
            try:
                input_text, output_text = map(str.strip, line.split('->', 1))
                processed_input = self.advanced_preprocess_text(input_text)
                X_texts.append(processed_input)
                
                features = self.extract_enhanced_features(output_text)
                for f in self.feature_names:
                    y_dict[f].append(features[f])
                
                valid_samples += 1
            except Exception as e:
                print(f"Satır işlenirken hata: {line[:50]}... - {e}")
                continue
        
        print(f"Toplam {valid_samples} geçerli örnek işlendi.")
        return X_texts, y_dict
    
    def get_best_model_for_feature(self, feature, X, y):
        """Her özellik için en iyi modeli seç"""
        
        # Sınıf dağılımını kontrol et
        class_counts = Counter(y)
        if len(class_counts) < 2:
            print(f"Uyarı: '{feature}' için yeterli çeşitlilik yok.")
            return RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Farklı modeller dene
        models_to_try = {
            'rf': RandomForestClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42)
        }
        
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gb': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        }
        
        best_score = -1
        best_model = None
        best_params = None
        
        for model_name, model in models_to_try.items():
            try:
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=3,  # 3-fold CV (daha hızlı)
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X, y)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
            except Exception as e:
                print(f"'{feature}' için {model_name} modeli eğitilirken hata: {e}")
                continue
        
        if best_model is None:
            # Fallback model
            best_model = RandomForestClassifier(n_estimators=50, random_state=42)
            best_model.fit(X, y)
            best_params = {}
        
        return best_model, best_params, best_score
    
    def train(self, data_file):
        print("=== Gelişmiş Model Eğitimi Başlıyor ===")
        print("Veri hazırlanıyor...")
        
        X, y_dict = self.prepare_training_data(data_file)
        
        if not X:
            print("Hata: Eğitim verisi bulunamadı!")
            return
        
        print(f"{len(X)} adet örnek bulundu.")
        print("Metinler vektörleştiriliyor...")
        
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"Toplam {X_vectorized.shape[1]} özellik çıkarıldı.")
        
        for feature in self.feature_names:
            print(f"\n--- '{feature}' özelliği eğitiliyor ---")
            
            # Label encoding
            le = LabelEncoder()
            y_feature = y_dict[feature]
            
            # Sınıf dağılımını göster
            unique_values = list(set(y_feature))
            print(f"Sınıflar: {unique_values}")
            
            if len(unique_values) < 2:
                print(f"Uyarı: '{feature}' için yeterli çeşitlilik yok, atlanıyor.")
                continue
            
            y_encoded = le.fit_transform(y_feature)
            
            # Özellik seçimi
            try:
                selector = SelectKBest(f_classif, k=min(500, X_vectorized.shape[1]))
                X_selected = selector.fit_transform(X_vectorized, y_encoded)
                self.feature_selectors[feature] = selector
                print(f"En iyi {X_selected.shape[1]} özellik seçildi.")
            except:
                X_selected = X_vectorized
                self.feature_selectors[feature] = None
            
            # En iyi modeli bul
            best_model, best_params, best_score = self.get_best_model_for_feature(
                feature, X_selected, y_encoded
            )
            
            self.models[feature] = best_model
            self.label_encoders[feature] = le
            self.best_params[feature] = best_params
            
            print(f"En iyi doğruluk: {best_score:.3f}")
            print(f"En iyi parametreler: {best_params}")
        
        print("\n=== Tüm modeller başarıyla eğitildi! ===")
    
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
                print(f"'{feature}' için tahmin hatası: {e}")
                predictions[feature] = 'none'
                confidences[feature] = 0.0
        
        return predictions, confidences
    
    def save_model(self, path):
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_selectors': self.feature_selectors,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'pattern_dict': self.pattern_dict
        }
        
        joblib.dump(model_data, path)
        print(f"Gelişmiş model '{path}' dosyasına kaydedildi.")
    
    def load_model(self, path):
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.feature_selectors = model_data.get('feature_selectors', {})
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params', {})
        self.pattern_dict = model_data.get('pattern_dict', {})
        print(f"Model '{path}' dosyasından yüklendi.")

if __name__ == "__main__":
    model = AdvancedPhoneRecommendationModel()
    
    # Eğitim
    model.train("training_data.txt")
    model.save_model("enhanced_phone_model.pkl")
    
    # Test örneği
    test_input = "ucuz android telefon oyun için 8gb ram"
    predictions, confidences = model.predict(test_input)
    
    print(f"\nTest girdisi: '{test_input}'")
    print("Tahminler:")
    for feature, pred in predictions.items():
        conf = confidences[feature]
        print(f"  {feature}: {pred} (güven: {conf:.2f})")
