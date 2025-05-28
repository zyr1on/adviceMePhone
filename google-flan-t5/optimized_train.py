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
from typing import Dict, List, Tuple
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def turkce_karakterleri_cevir(text):
    """Türkçe karakterleri İngilizce'ye çevir"""
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
    """Geliştirilmiş metin ön işleme"""
    text = text.lower().strip()
    text = turkce_karakterleri_cevir(text)
    # Noktalama işaretlerini normalize et
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class SmartAugmentation:
    """Akıllı veri augmentation sınıfı"""
    
    def __init__(self):
        self.synonyms = {
            'telefon': ['telefon', 'cep telefonu', 'akilli telefon', 'mobil', 'cihaz'],
            'oner': ['oner', 'tavsiye et', 'onerir misin', 'goster', 'soyle', 'ver', 'bul'],
            'iyi': ['iyi', 'kaliteli', 'guzel', 'mukemmel', 'basarili', 'etkili'],
            'ucuz': ['ucuz', 'uygun fiyatli', 'ekonomik', 'hesapli'],
            'oyun': ['oyun', 'game', 'gaming', 'mobil oyun'],
            'kamera': ['kamera', 'fotograf', 'resim', 'fotografcilik'],
            'batarya': ['batarya', 'pil', 'sarj'],
            'hizli': ['hizli', 'performansli', 'guclu'],
            'android': ['android', 'samsung', 'xiaomi'],
            'iphone': ['iphone', 'apple', 'ios']
        }
        
        self.brand_variations = {
            'samsung': ['samsung', 'galaxy'],
            'apple': ['apple', 'iphone'],
            'xiaomi': ['xiaomi', 'redmi', 'poco'],
            'huawei': ['huawei', 'honor'],
            'oppo': ['oppo', 'realme']
        }
        
    def augment_input(self, text: str, num_variations: int = 2) -> List[str]:
        """Akıllı augmentation ile varyasyon oluştur"""
        variations = [text]  # Orijinal metni dahil et
        
        for _ in range(num_variations):
            augmented = text
            
            # Sinonim değişimi
            for word, syns in self.synonyms.items():
                if word in augmented and random.random() < 0.4:
                    augmented = augmented.replace(word, random.choice(syns))
            
            # Marka varyasyonları
            for brand, vars in self.brand_variations.items():
                if brand in augmented and random.random() < 0.3:
                    augmented = augmented.replace(brand, random.choice(vars))
            
            # Kelime ekleme
            if random.random() < 0.2:
                additions = ['lutfen', 'acil', 'en iyi', 'oneriniz nedir']
                augmented = f"{random.choice(additions)} {augmented}"
            
            if augmented != text and augmented not in variations:
                variations.append(augmented)
        
        return variations

class OptimizedPhoneDataset(Dataset):
    """Optimize edilmiş dataset sınıfı"""
    
    def __init__(self, inputs: List[str], targets: List[str], tokenizer, 
                 max_input_length: int = 128, max_target_length: int = 256, 
                 augment_data: bool = True, augment_ratio: float = 0.4):
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.augmenter = SmartAugmentation()
        
        # Veri augmentation - akıllı şekilde
        if augment_data and len(inputs) > 0:
            augmented_inputs, augmented_targets = self._smart_augment(
                inputs, targets, augment_ratio
            )
            self.inputs = augmented_inputs
            self.targets = augmented_targets
        else:
            self.inputs = inputs
            self.targets = targets
            
        logger.info(f"Dataset oluşturuldu: {len(self.inputs)} örnek (orijinal: {len(inputs)})")

    def _smart_augment(self, inputs: List[str], targets: List[str], 
                      augment_ratio: float) -> Tuple[List[str], List[str]]:
        """Akıllı augmentation stratejisi"""
        augmented_inputs = []
        augmented_targets = []
        
        # Önce tüm orijinal veriyi ekle
        augmented_inputs.extend(inputs)
        augmented_targets.extend(targets)
        
        # Veri dağılımını analiz et
        target_distribution = Counter(targets)
        rare_threshold = len(inputs) * 0.05  # %5'ten az olan nadir örnekler
        
        for inp, tgt in zip(inputs, targets):
            # Nadir örnekleri daha çok augment et
            is_rare = target_distribution[tgt] < rare_threshold
            augment_prob = 0.7 if is_rare else augment_ratio
            
            if random.random() < augment_prob:
                variations = self.augmenter.augment_input(inp, num_variations=2 if is_rare else 1)
                for var in variations[1:]:  # İlkini zaten ekledik
                    augmented_inputs.append(var)
                    augmented_targets.append(tgt)
        
        return augmented_inputs, augmented_targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"telefon analizi: {self.inputs[idx]} -> ozellikler:"
        target_text = self.targets[idx]

        # Tokenization - daha efficient
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

class FocalLoss(torch.nn.Module):
    """Dengesiz veri setleri için Focal Loss"""
    
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(pred_valid, target_valid, reduction='none')
        
        # Focal loss computation
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class OptimizedTrainer(Trainer):
    """Optimize edilmiş trainer sınıfı"""
    
    def __init__(self, *args, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if hasattr(self, 'loss_fn'):
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs, **kwargs)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            loss = self.loss_fn(logits_flat, labels_flat)
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

class EnhancedPhoneT5Model:
    """Geliştirilmiş T5 telefon modeli"""
    
    def __init__(self, model_name='google/flan-t5-base'):  # FLAN-T5 daha iyi
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Özel tokenlar - daha az ve daha anlamlı
        special_tokens = {
            'additional_special_tokens': [
                '<feature>', '<value>', '<price_range>', '<brand_name>',
                '<storage_size>', '<ram_size>', '<camera_mp>', '<battery_mah>'
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_stats = {}
        self.feature_keys = ['price', 'brand', 'os', 'usage', 'ram', 
                           'storage', 'battery', 'camera', 'screen']

    def validate_and_clean_data(self, inputs: List[str], targets: List[str]) -> Tuple[List[str], List[str]]:
        """Veri doğrulama ve temizleme"""
        cleaned_inputs = []
        cleaned_targets = []
        
        for inp, tgt in zip(inputs, targets):
            try:
                # Input kontrolü
                if not inp.strip() or len(inp.strip()) < 3:
                    continue
                
                # Target yapısını kontrol et
                if ';' not in tgt or ':' not in tgt:
                    continue
                
                # Feature parsing test
                features = {}
                pairs = tgt.split(';')
                valid_pairs = 0
                
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key in self.feature_keys:
                            features[key] = value
                            valid_pairs += 1
                
                # En az 5 valid feature olmalı
                if valid_pairs >= 5:
                    # Eksik featureları 'none' ile doldur
                    for key in self.feature_keys:
                        if key not in features:
                            features[key] = 'none'
                    
                    # Yeniden yapılandır
                    structured_target = '; '.join([f"{k}:{v}" for k, v in features.items()])
                    
                    cleaned_inputs.append(preprocess_text(inp))
                    cleaned_targets.append(structured_target)
                    
            except Exception as e:
                logger.warning(f"Hatalı veri atlandı: {tgt[:50]}... - Hata: {e}")
                continue
        
        logger.info(f"Veri temizleme: {len(inputs)} -> {len(cleaned_inputs)} örnek")
        return cleaned_inputs, cleaned_targets

    def analyze_data_distribution(self, targets: List[str]) -> Dict:
        """Veri dağılımını detaylı analiz et"""
        feature_stats = {}
        value_distributions = {}
        
        for target in targets:
            pairs = target.split(';')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key not in feature_stats:
                        feature_stats[key] = Counter()
                        value_distributions[key] = []
                    
                    feature_stats[key][value] += 1
                    value_distributions[key].append(value)
        
        # İstatistikleri hesapla
        analysis = {
            'total_samples': len(targets),
            'feature_distributions': {},
            'imbalance_ratios': {}
        }
        
        for feature, counter in feature_stats.items():
            total = sum(counter.values())
            analysis['feature_distributions'][feature] = dict(counter.most_common())
            
            # Dengesizlik oranını hesapla
            if total > 0:
                max_count = max(counter.values())
                min_count = min(counter.values())
                analysis['imbalance_ratios'][feature] = max_count / min_count if min_count > 0 else float('inf')
        
        return analysis

    def load_data(self, data_file: str) -> Tuple[List[str], List[str]]:
        """Optimize edilmiş veri yükleme"""
        logger.info(f"Veri dosyası okunuyor: {data_file}")
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"HATA: {data_file} dosyası bulunamadı!")
            return [], []

        inputs = []
        targets = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or " -> " not in line:
                continue
                
            try:
                input_text, output_text = line.split(" -> ", 1)
                if input_text.strip() and output_text.strip():
                    inputs.append(input_text.strip())
                    targets.append(output_text.strip())
            except Exception as e:
                logger.warning(f"Satır {line_num} işlenemedi: {e}")
                continue

        logger.info(f"Ham veri yüklendi: {len(inputs)} örnek")
        
        # Veri temizleme ve doğrulama
        inputs, targets = self.validate_and_clean_data(inputs, targets)
        
        # Veri analizi
        self.data_stats = self.analyze_data_distribution(targets)
        
        return inputs, targets

    def train(self, data_file: str, output_dir: str = './optimized_phone_t5_model',
              epochs: int = 15, batch_size: int = 8, learning_rate: float = 3e-5):
        """Optimize edilmiş eğitim"""
        
        inputs, targets = self.load_data(data_file)
        if len(inputs) == 0:
            logger.error("HATA: Hiç veri yüklenemedi!")
            return

        # Stratified split - daha dengeli dağılım
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.15, random_state=42, 
            shuffle=True, stratify=None  # Çok kategorili olduğu için stratify kapalı
        )

        logger.info(f"Eğitim seti: {len(train_inputs)} örnek")
        logger.info(f"Validasyon seti: {len(val_inputs)} örnek")

        # Dataset oluştur
        train_dataset = OptimizedPhoneDataset(
            train_inputs, train_targets, self.tokenizer,
            max_input_length=128, max_target_length=200,
            augment_data=True, augment_ratio=0.5
        )
        
        val_dataset = OptimizedPhoneDataset(
            val_inputs, val_targets, self.tokenizer,
            max_input_length=128, max_target_length=200,
            augment_data=False
        )

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        logger.info(f"Kullanılan cihaz: {device}")

        # Optimize edilmiş training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,  # Eval için daha büyük batch
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=20,
            eval_strategy='steps',
            eval_steps=100,
            save_strategy='steps', 
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to="none",
            seed=42,
            remove_unused_columns=False,
            optim="adamw_torch",
            adam_epsilon=1e-8,
            adam_beta1=0.9,
            adam_beta2=0.999,
        )

        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        # Trainer oluştur
        trainer = OptimizedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            use_focal_loss=False  # 1000 örnek için focal loss gerekli olmayabilir
        )

        logger.info("Model eğitimi başlıyor...")
        trainer.train()

        # Model kaydet
        logger.info("Model kaydediliyor...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # İstatistikleri kaydet
        with open(os.path.join(output_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(self.data_stats, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Model {output_dir} klasörüne kaydedildi!")

        # Test
        self._evaluate_model(val_inputs[:10], val_targets[:10])

    def _evaluate_model(self, test_inputs: List[str], test_targets: List[str]):
        """Detaylı model değerlendirme"""
        logger.info("\n--- Model Değerlendirme ---")
        
        correct_predictions = 0
        total_features = 0
        feature_accuracies = {key: {'correct': 0, 'total': 0} for key in self.feature_keys}
        
        for i, (inp, expected) in enumerate(zip(test_inputs, test_targets)):
            predicted = self.generate_response(inp)
            
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Girdi: {inp}")
            logger.info(f"Beklenen: {expected}")
            logger.info(f"Tahmin: {predicted}")
            
            # Feature-level doğruluk
            expected_features = self._parse_structured_output(expected)
            predicted_features = self._parse_structured_output(predicted)
            
            for key in self.feature_keys:
                expected_val = expected_features.get(key, 'none')
                predicted_val = predicted_features.get(key, 'none')
                
                feature_accuracies[key]['total'] += 1
                if expected_val == predicted_val:
                    feature_accuracies[key]['correct'] += 1
                    correct_predictions += 1
                
                total_features += 1
        
        # Genel sonuçlar
        overall_accuracy = correct_predictions / total_features if total_features > 0 else 0
        logger.info(f"\nGenel doğruluk: {overall_accuracy:.2%}")
        
        # Feature-level sonuçlar
        logger.info("\nÖzellik bazlı doğruluk:")
        for key, stats in feature_accuracies.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                logger.info(f"  {key}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    def _parse_structured_output(self, output_text: str) -> Dict[str, str]:
        """Yapılandırılmış çıktıyı parse et"""
        features = {key: 'none' for key in self.feature_keys}
        
        try:
            pairs = output_text.split(';')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in features:
                        features[key] = value
        except Exception as e:
            logger.warning(f"Parse hatası: {e}")
        
        return features

    def generate_response(self, input_text: str, max_length: int = 200, 
                         num_beams: int = 4, temperature: float = 0.8) -> str:
        """Optimize edilmiş response generation"""
        
        input_text = preprocess_text(input_text)
        formatted_input = f"telefon analizi: {input_text} -> ozellikler:"

        input_ids = self.tokenizer.encode(
            formatted_input,
            return_tensors='pt',
            max_length=128,
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
                do_sample=True,
                top_p=0.95,
                top_k=50,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                length_penalty=1.0,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def load_trained_model(self, model_path: str):
        """Eğitilmiş modeli yükle"""
        logger.info(f"Model yükleniyor: {model_path}")
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)

            stats_path = os.path.join(model_path, 'data_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.data_stats = json.load(f)

            logger.info("Model başarıyla yüklendi!")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")

def main():
    """Ana fonksiyon"""
    print("Optimize Edilmiş T5 Telefon Önerisi Modeli")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    print(f"CUDA kullanılabilir: {torch.cuda.is_available()}")

    try:
        # Model oluştur - FLAN-T5 daha iyi instruction following için
        model = EnhancedPhoneT5Model('google/flan-t5-base')
        
        # Optimize edilmiş eğitim
        model.train(
            data_file='training_data.txt',
            epochs=20,  # 1000 örnek için yeterli
            batch_size=8,  # GPU memory'ye uygun
            learning_rate=2e-5  # FLAN-T5 için optimize
        )

        print("\nEğitim tamamlandı!")
        
        # Kapsamlı test
        test_queries = [
            "samsung bataryası iyi olan",
            "telefon öner",
            "oyun için telefon", 
            "15000 tl altinda android",
            "kamerasi iyi olan iphone",
            "ucuz telefon 8gb ram",
            "256 gb hafiza fotograf",
            "apple ios telefon"
        ]

        print("\n" + "="*60)
        print("KAPSAMLI TEST")
        print("="*60)

        for query in test_queries:
            print(f"\nTest: {query}")
            result = model.generate_response(query)
            print(f"Sonuç: {result}")
            
            # Parse et ve göster
            features = model._parse_structured_output(result)
            print("Özellikler:")
            for key, value in features.items():
                if value != 'none':
                    print(f"  {key}: {value}")
            print("-" * 40)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
