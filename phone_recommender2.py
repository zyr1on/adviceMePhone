try:
	import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
    import pandas as pd
    import re
    import csv
except ImportError as e:
	print("\n[!] Gerekli modüller yüklenemedi:")
	print(f"    -> {e}")
	print("\nLütfen gerekli bağımlılıkları yüklemek için aşağıdaki komutu çalıştırın:")
	print("    python3 install.py\n")

class PhonePredictor2:
    def __init__(self, model_path="model.pt", data_path="phones.csv", labels_path="labels.txt"):
        self.model_path = model_path
        self.data_path = data_path
        self.labels_path = labels_path
        self.model = None
        self.tokenizer = None
        self.phones_df = None
        self.labels = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_data()
        self.load_labels()

    def load_model(self):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            with open(self.labels_path, "r", encoding="utf-8") as f:
                raw = f.read()
            self.labels = sorted(set([l.strip() for l in re.findall(r"-(.+)", raw)]))
            config = DistilBertConfig(
                num_labels=len(self.labels),
                problem_type="multi_label_classification"
            )
            self.model = DistilBertForSequenceClassification(config)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise Exception(f"Model yüklenirken hata: {e}")

    def load_data(self):
        try:
            self.phones_df = pd.read_csv(self.data_path, quoting=csv.QUOTE_NONNUMERIC)
            self.phones_df.columns = [col.lower().strip() for col in self.phones_df.columns]
            self.phones_df = self.phones_df.replace('noneP', 'none')
            self.phones_df = self.phones_df.fillna('none')
            self.phones_df['price'] = pd.to_numeric(self.phones_df['price'], errors='coerce').fillna(0.0)
            self.phones_df['ram'] = pd.to_numeric(self.phones_df['ram'], errors='coerce').fillna(0.0)
            for col in ['os', 'storage', 'camera', 'battery', 'screen', 'usage', 'brand', 'link']:
                if col in self.phones_df.columns:
                    self.phones_df[col] = self.phones_df[col].astype(str).str.lower().str.strip()
        except Exception as e:
            raise Exception(f"Veri seti yüklenirken hata: {e}")

    def load_labels(self):
        try:
            self.id2label = {i: label for i, label in enumerate(self.labels)}
        except Exception as e:
            raise Exception(f"Etiketler yüklenirken hata: {e}")

    def add_label_with_conflict_check(self, predicted_labels, new_label):
        category = new_label.split(':')[0]
        for label in list(predicted_labels):
            if label.startswith(category + ':'):
                predicted_labels.remove(label)
        if new_label not in predicted_labels:
            predicted_labels.append(new_label)

    def predict(self, input_text, threshold=0.41):
        encoded = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0]
        
        predicted_labels = []
        for i, prob in enumerate(probs):
            if prob > threshold:
                tag = self.id2label[i]
                self.add_label_with_conflict_check(predicted_labels, tag)
        
        predictions = {}
        for label in predicted_labels:
            category, value = label.split(':')
            predictions[category] = value
        
        confidences = {label: float(probs[i]) for i, label in enumerate(self.labels) if label in predicted_labels}
        return [predictions], [confidences], [f"{input_text}: " + "; ".join(f"{k}:{v}" for k, v in predictions.items())], [{}]

    def recommend_phones(self, predictions_list, confidences_list, extracted_values_list, top_n=None):
        filtered_df = self.phones_df.copy()
        applied_filters = []
        
        for predictions in predictions_list:
            for feature, value in predictions.items():
                if value != 'none' and feature in filtered_df.columns and feature not in ['price', 'ram']:
                    try:
                        filtered_df = filtered_df[filtered_df[feature].apply(lambda x: str(x).lower()) == value]
                        applied_filters.append(f"{feature}={value}")
                    except:
                        continue
        
        try:
            filtered_df = filtered_df.sort_values(by='price')
        except:
            filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce').fillna(0.0)
            filtered_df = filtered_df.sort_values(by='price')
        
        recommendations = filtered_df if top_n is None else filtered_df.head(top_n)
        return recommendations
