import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import re
import json
import numpy as np
from collections import Counter
import random
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

def turkce_karakterleri_cevir(text):
    cevirme_dict = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ü': 'u', 'Ü': 'U'
    }
    for tr_char, en_char in cevirme_dict.items():
        text = text.replace(tr_char, en_char)
    return text

def preprocess_text(text):
    text = text.lower().strip()
    text = turkce_karakterleri_cevir(text)
    # Daha iyi normalizasyon
    text = re.sub(r'\s+', ' ', text)  # Birden fazla boşluğu tek boşluğa çevir
    return text

# Veri augmentation fonksiyonları
def augment_input(text):
    """Input varyasyonları oluştur"""
    variations = []
    
    # Orijinal metni ekle
    variations.append(text)
    
    # Sinonim değişimleri
    synonyms = {
        'telefon': ['telefon', 'cep telefonu', 'akilli telefon',"mobil","cihaz"],
        'oner': ['oner', 'onerisi', 'onerir misin', 'tavsiye et',"goster","soyle","ver"],
        'iyi': ['iyi', 'kaliteli', 'guzel', 'mukemmel'],
        'ucuz': ['ucuz', 'uygun fiyatli', 'ekonomik'],
        'oyun': ['oyun', 'game', 'gaming'],
        'kamera': ['kamera', 'fotograf', 'resim']
    }
    
    # Sinonim değişimi
    augmented = text
    for word, syns in synonyms.items():
        if word in augmented:
            augmented = augmented.replace(word, random.choice(syns))
    if augmented != text:
        variations.append(augmented)
    
    # Kelime sırası değişimi (basit)
    words = text.split()
    if len(words) > 2:
        shuffled = words.copy()
        random.shuffle(shuffled)
        variations.append(' '.join(shuffled))
    
    return variations

def create_structured_prompt(input_text):
    """Daha yapılandırılmış prompt oluştur"""
    return f"telefon ozellik analizi: {input_text} -> ozellikler:"

class ImprovedPhoneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_length=128, max_target_length=256, 
                 augment_data=True):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Veri augmentation
        if augment_data:
            augmented_inputs = []
            augmented_targets = []
            
            for inp, tgt in zip(inputs, targets):
                # Orijinal veriyi ekle
                augmented_inputs.append(inp)
                augmented_targets.append(tgt)
                
                # %30 ihtimalle augmentation yap
                if random.random() < 0.3:
                    variations = augment_input(inp)
                    for var in variations[1:]:  # İlkini zaten ekledik
                        augmented_inputs.append(var)
                        augmented_targets.append(tgt)
            
            self.inputs = augmented_inputs
            self.targets = augmented_targets
        else:
            self.inputs = inputs
            self.targets = targets
        
        print(f"Dataset oluşturuldu: {len(self.inputs)} örnek")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = create_structured_prompt(str(self.inputs[idx]))
        target_text = str(self.targets[idx])

        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# Özel loss function - CUDA uyumlu
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # Geçerli hedefleri filtrele (-100 ignore)
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Sadece geçerli pozisyonları kullan
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        n_class = pred_valid.size(-1)
        
        # One-hot encoding güvenli şekilde
        target_valid = target_valid.clamp(0, n_class-1)  # Güvenlik için clamp
        one_hot = torch.zeros_like(pred_valid).scatter(-1, target_valid.unsqueeze(-1), 1)
        
        # Label smoothing uygula
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        
        log_prb = F.log_softmax(pred_valid, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1).mean()
        
        return loss

class CustomTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingLoss(smoothing=label_smoothing)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Yeni Transformers versiyonu için uyumlu compute_loss metodu"""
        if hasattr(self, 'loss_fn'):
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs, **kwargs)
            
            # Labels'ı inputs'tan çıkarmıyoruz, kopyalıyoruz
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Reshape for loss calculation - güvenli şekilde
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            # CUDA güvenliği için device kontrolü
            if logits_flat.device != labels_flat.device:
                labels_flat = labels_flat.to(logits_flat.device)
            
            loss = self.loss_fn(logits_flat, labels_flat)
            
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

class EnhancedPhoneT5Model:
    def __init__(self, model_name='t5-small'):  # Küçük modelle başla
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Special tokens ekle
        special_tokens = {
            'additional_special_tokens': [
                '<price>', '<brand>', '<os>', '<usage>', '<ram>', 
                '<storage>', '<battery>', '<camera>', '<screen>',
                '<android>', '<ios>', '<good>', '<bad>', '<any>', '<none>'
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_stats = {}

    def parse_structured_output(self, output_text):
        """Yapılandırılmış çıktıyı parse et"""
        features = {
            'price': 'any', 'brand': 'none', 'os': 'none', 'usage': 'none',
            'ram': 'none', 'storage': 'none', 'battery': 'none', 
            'camera': 'none', 'screen': 'none'
        }
        
        # Parse the structured output
        pairs = output_text.split(';')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in features:
                    features[key] = value
        
        return features

    def validate_output_structure(self, inputs, targets):
        """Çıktı yapısını doğrula ve düzelt"""
        cleaned_inputs = []
        cleaned_targets = []
        
        required_keys = ['price', 'brand', 'os', 'usage', 'ram', 'storage', 'battery', 'camera', 'screen']
        
        for inp, tgt in zip(inputs, targets):
            try:
                # Target'ı parse et
                features = {}
                pairs = tgt.split(';')
                
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        features[key] = value
                
                # Eksik anahtarları 'none' ile doldur
                for key in required_keys:
                    if key not in features:
                        features[key] = 'none'
                
                # Yeniden yapılandır
                structured_target = '; '.join([f"{k}:{v}" for k, v in features.items()])
                
                cleaned_inputs.append(inp)
                cleaned_targets.append(structured_target)
                
            except Exception as e:
                print(f"Hatalı veri atlandı: {tgt} - Hata: {e}")
                continue
        
        return cleaned_inputs, cleaned_targets

    def analyze_data(self, inputs, targets):
        """Geliştirilmiş veri analizi"""
        print("\n--- Gelişmiş Veri Analizi ---")
        
        # Temel istatistikler
        input_lengths = [len(inp.split()) for inp in inputs]
        target_lengths = [len(tgt.split()) for tgt in targets]

        print(f"Toplam örnek sayısı: {len(inputs)}")
        print(f"Input ortalama kelime sayısı: {np.mean(input_lengths):.1f}")
        print(f"Target ortalama kelime sayısı: {np.mean(target_lengths):.1f}")

        # Özellik dağılımı analizi
        feature_stats = {}
        for target in targets:
            try:
                pairs = target.split(';')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key not in feature_stats:
                            feature_stats[key] = Counter()
                        feature_stats[key][value] += 1
            except:
                continue

        print("\n--- Özellik Dağılımları ---")
        for feature, counter in feature_stats.items():
            print(f"{feature}:")
            for value, count in counter.most_common(5):
                print(f"  {value}: {count}")
            print()

        return {
            'input_lengths': input_lengths,
            'target_lengths': target_lengths,
            'avg_input_len': np.mean(input_lengths),
            'avg_target_len': np.mean(target_lengths),
            'feature_stats': feature_stats
        }

    def load_data(self, data_file):
        inputs = []
        targets = []

        print(f"Veri dosyası okunuyor: {data_file}")

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"HATA: {data_file} dosyası bulunamadı!")
            return [], []

        valid_lines = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if " -> " not in line:
                continue
            try:
                input_text, output_text = line.split(" -> ", 1)
                if not input_text.strip() or not output_text.strip():
                    continue
                
                # Preprocessing
                input_text = preprocess_text(input_text.strip())
                output_text = output_text.strip()
                
                inputs.append(input_text)
                targets.append(output_text)
                valid_lines += 1
            except Exception as e:
                continue

        print(f"Ham veri: {valid_lines} örnek yüklendi")
        
        # Yapılandırılmış çıktı doğrulama
        inputs, targets = self.validate_output_structure(inputs, targets)
        
        # Veri analizi
        self.data_stats = self.analyze_data(inputs, targets)

        return inputs, targets

    def train(self, data_file, output_dir='./enhanced_phone_t5_model', 
              epochs=20, batch_size=8, learning_rate=5e-5):
        
        inputs, targets = self.load_data(data_file)
        if len(inputs) == 0:
            print("HATA: Hiç veri yüklenemedi!")
            return

        # Train-validation split
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42, shuffle=True
        )

        print(f"Eğitim seti: {len(train_inputs)} örnek")
        print(f"Validasyon seti: {len(val_inputs)} örnek")

        # Dataset oluştur - Optimal uzunluklar
        train_dataset = ImprovedPhoneDataset(
            train_inputs, train_targets, self.tokenizer, 
            max_input_length=192, max_target_length=320, 
            augment_data=True
        )
        
        val_dataset = ImprovedPhoneDataset(
            val_inputs, val_targets, self.tokenizer,
            max_input_length=192, max_target_length=320,
            augment_data=False
        )

        # GPU optimizasyonu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Training arguments - 800 örnek için optimize
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,  # 800 örnek için biraz yüksek
            warmup_ratio=0.1,
            warmup_steps=100,
            weight_decay=0.01,   # Daha güçlü regularization
            logging_dir='./logs',
            logging_steps=25,    # Daha sık log
            eval_strategy='steps',
            eval_steps=50,       # Daha sık evaluation
            save_strategy='steps',
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            gradient_accumulation_steps=4,  # Daha büyük effective batch
            max_grad_norm=1.0,
            lr_scheduler_type="cosine_with_restarts",
            report_to="none",
            seed=42,
        )

        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        # Custom trainer with safer settings
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            label_smoothing=0.05  # Daha düşük label smoothing
        )

        print("Model eğitimi başlıyor...")
        trainer.train()

        # Model kaydet
        print("Model kaydediliyor...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Stats kaydet
        with open(os.path.join(output_dir, 'data_stats.json'), 'w') as f:
            json.dump(self.data_stats, f, indent=2, default=str)

        print(f"Model {output_dir} klasörüne kaydedildi!")

        # Test
        self.evaluate_model(val_inputs[:5], val_targets[:5])

    def evaluate_model(self, test_inputs, test_targets):
        """Model performansını değerlendir"""
        print("\n--- Model Değerlendirme ---")
        
        correct_predictions = 0
        total_features = 0
        
        for i, (inp, expected) in enumerate(zip(test_inputs, test_targets)):
            predicted = self.generate_response(inp)
            
            print(f"\nTest {i+1}:")
            print(f"Girdi: {inp}")
            print(f"Beklenen: {expected}")
            print(f"Tahmin: {predicted}")
            
            # Doğruluk hesapla
            expected_features = self.parse_structured_output(expected)
            predicted_features = self.parse_structured_output(predicted)
            
            feature_matches = 0
            for key in expected_features:
                if key in predicted_features and expected_features[key] == predicted_features[key]:
                    feature_matches += 1
            
            accuracy = feature_matches / len(expected_features)
            print(f"Özellik doğruluğu: {accuracy:.2%}")
            
            correct_predictions += feature_matches
            total_features += len(expected_features)
        
        overall_accuracy = correct_predictions / total_features if total_features > 0 else 0
        print(f"\nGenel doğruluk: {overall_accuracy:.2%}")

    def generate_response(self, input_text, max_length=256, num_beams=5, 
                         temperature=0.7, do_sample=True, top_p=0.9):
        
        input_text = preprocess_text(input_text)
        formatted_input = create_structured_prompt(input_text)

        input_ids = self.tokenizer.encode(
            formatted_input,
            return_tensors='pt',
            max_length=192,  # Input için optimal uzunluk
            truncation=True
        )

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                length_penalty=1.0,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def load_trained_model(self, model_path):
        print(f"Model yükleniyor: {model_path}")
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)

            stats_path = os.path.join(model_path, 'data_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.data_stats = json.load(f)

            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

def main():
    print("Geliştirilmiş T5 Telefon Önerisi Modeli")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    try:
        # Model oluştur
        model = EnhancedPhoneT5Model('t5-base')  # Küçük modelle başla
        
        # Eğitim
        model.train(
            data_file='training_data.txt',
            epochs=25,
            batch_size=4,
            learning_rate=1e-5
        )

        print("\nEğitim tamamlandı!")
        
        # Kapsamlı test
        test_queries = [
            "samsung bataryası iyi olan",
            "telefon öner",
            "oyun için telefon",
            "15000 tl altinda android",
            "kamerasi iyi olan iphone",
            "ucuz telefon 8gb ram"
        ]

        print("\n" + "="*60)
        print("KAPSAMLI TEST")
        print("="*60)

        for query in test_queries:
            print(f"\nTest: {query}")
            result = model.generate_response(query)
            print(f"Sonuç: {result}")
            
            # Parse et ve göster
            features = model.parse_structured_output(result)
            print("Özellikler:")
            for key, value in features.items():
                print(f"  {key}: {value}")
            print("-" * 40)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
