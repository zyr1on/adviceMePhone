import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import re
import json
import numpy as np
from collections import Counter

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
    # Daha kapsamlı temizlik
    text = text.lower().strip()
    text = turkce_karakterleri_cevir(text)
    return text

def normalize_price(text):
    return text

def normalize_specs(text):
   return text

class PhoneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_length=256, max_target_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Daha spesifik prefix kullan
        input_text = f"telefon onerisi: {str(self.inputs[idx])}"
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

class PhoneT5Model:
    def __init__(self, model_name='t5-base'):  # 40GB VRAM ile t5-base kullanabilirsin
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Veri istatistikleri için
        self.data_stats = {}

    def analyze_data(self, inputs, targets):
        """Veri kalitesini analiz et"""
        print("\n--- Veri Analizi ---")
        
        # Input uzunluk analizi
        input_lengths = [len(inp.split()) for inp in inputs]
        target_lengths = [len(tgt.split()) for tgt in targets]
        
        print(f"Input ortalama kelime sayısı: {np.mean(input_lengths):.1f}")
        print(f"Target ortalama kelime sayısı: {np.mean(target_lengths):.1f}")
        
        # Tekrar eden pattern'leri bul
        common_inputs = Counter(inputs).most_common(5)
        common_targets = Counter(targets).most_common(5)
        
        print("\nEn sık tekrar eden input'lar:")
        for inp, count in common_inputs:
            if count > 1:
                print(f"  '{inp}' - {count} kez")
        
        print("\nEn sık tekrar eden target'lar:")
        for tgt, count in common_targets:
            if count > 1:
                print(f"  '{tgt}' - {count} kez")
        
        return {
            'input_lengths': input_lengths,
            'target_lengths': target_lengths,
            'avg_input_len': np.mean(input_lengths),
            'avg_target_len': np.mean(target_lengths)
        }

    def clean_and_validate_data(self, inputs, targets):
        """Veri temizleme ve validasyon"""
        cleaned_inputs = []
        cleaned_targets = []
        
        for inp, tgt in zip(inputs, targets):
            # Daha agresif temizlik
            cleaned_inp = normalize_price(normalize_specs(preprocess_text(inp)))
            cleaned_tgt = normalize_price(normalize_specs(preprocess_text(tgt)))
            
            # Çok kısa veya çok uzun verileri filtrele
            if (len(cleaned_inp.split()) >= 3 and len(cleaned_inp.split()) <= 20 and
                len(cleaned_tgt.split()) >= 3 and len(cleaned_tgt.split()) <= 30):
                cleaned_inputs.append(cleaned_inp)
                cleaned_targets.append(cleaned_tgt)
        
        print(f"Veri temizleme: {len(inputs)} -> {len(cleaned_inputs)} örnek")
        return cleaned_inputs, cleaned_targets

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
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
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
                inputs.append(input_text.strip())
                targets.append(output_text.strip())
                valid_lines += 1
            except Exception as e:
                continue

        print(f"Ham veri: {valid_lines} örnek yüklendi")
        
        # Veri temizleme ve validasyon
        inputs, targets = self.clean_and_validate_data(inputs, targets)
        
        # Veri analizi
        self.data_stats = self.analyze_data(inputs, targets)
        
        # Örnekleri göster
        if len(inputs) > 0:
            print("\n--- Temizlenmiş Örnek Veriler ---")
            for i in range(min(3, len(inputs))):
                print(f"Girdi: {inputs[i]}")
                print(f"Çıktı: {targets[i]}")
                print("-" * 50)

        return inputs, targets

    def train(self, data_file, output_dir='./phone_t5_model_optimized', epochs=15, batch_size=32, learning_rate=3e-5):
        inputs, targets = self.load_data(data_file)
        if len(inputs) == 0:
            print("HATA: Hiç veri yüklenemedi!")
            return

        # Stratified split için benzer uzunluktaki verileri grupla
        input_lengths = [len(inp.split()) for inp in inputs]
        length_groups = ['short' if l <= 5 else 'medium' if l <= 10 else 'long' for l in input_lengths]
        
        # Test split oranını veri miktarına göre ayarla
        test_size = 0.2 if len(inputs) >= 100 else 0.15
        
        try:
            train_inputs, val_inputs, train_targets, val_targets = train_test_split(
                inputs, targets, test_size=test_size, random_state=42, 
                shuffle=True, stratify=length_groups
            )
        except:
            # Stratify başarısız olursa normal split
            train_inputs, val_inputs, train_targets, val_targets = train_test_split(
                inputs, targets, test_size=test_size, random_state=42, shuffle=True
            )

        # GPU kapasitesine göre batch size ayarlama
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU Bellek: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 40:  # Yüksek kapasiteli GPU
                recommended_batch = 16
                gradient_accum_steps = 1
                max_input_len = 256
                max_target_len = 512
            elif gpu_memory >= 24:  # Orta-yüksek kapasiteli GPU
                recommended_batch = 16
                gradient_accum_steps = 2
                max_input_len = 128
                max_target_len = 256
            elif gpu_memory >= 12:  # Orta kapasiteli GPU
                recommended_batch = 8
                gradient_accum_steps = 4
                max_input_len = 128
                max_target_len = 256
            else:  # Düşük kapasiteli GPU
                recommended_batch = 4
                gradient_accum_steps = 8
                max_input_len = 96
                max_target_len = 192
                
            if batch_size != recommended_batch:
                print(f"Önerilen batch size: {recommended_batch} (mevcut: {batch_size})")
        else:
            gradient_accum_steps = 2
            max_input_len = 256
            max_target_len = 512
        avg_input_len = int(self.data_stats.get('avg_input_len', 10))
        avg_target_len = int(self.data_stats.get('avg_target_len', 15))
        
        # GPU kapasitesine göre ayarlanan değerleri kullan
        max_input_length = min(max_input_len, max(258, avg_input_len * 8))
        max_target_length = min(max_target_len, max(512, avg_target_len * 10))
        
        print(f"Max input length: {max_input_length}, Max target length: {max_target_length}")

        train_dataset = PhoneDataset(train_inputs, train_targets, self.tokenizer, 
                                   max_input_length, max_target_length)
        val_dataset = PhoneDataset(val_inputs, val_targets, self.tokenizer,
                                 max_input_length, max_target_length)

        # Optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,  # Daha yumuşak warmup
            weight_decay=0.1,   # Daha güçlü regularization
            logging_dir='./logs',
            logging_steps=max(1, len(train_inputs) // (batch_size * 10)),
            eval_strategy='steps',
            eval_steps=max(50, len(train_inputs) // (batch_size * 5)),
            save_strategy='steps',
            save_steps=max(50, len(train_inputs) // (batch_size * 5)),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            gradient_accumulation_steps=gradient_accum_steps,  # GPU kapasitesine göre ayarlandı
            max_grad_norm=1.0,  # Gradient clipping
            report_to="none",
            seed=42,
            lr_scheduler_type="cosine",  # Cosine learning rate scheduler
        )
        
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Model eğitimi başlıyor...")
        print(f"Cihaz: {next(self.model.parameters()).device}")

        trainer.train()

        print("Model kaydediliyor...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Data stats'ı da kaydet
        with open(os.path.join(output_dir, 'data_stats.json'), 'w') as f:
            json.dump(self.data_stats, f, indent=2)

        print(f"Model {output_dir} klasörüne kaydedildi!")

        # Kapsamlı test
        print("\n--- Test Sonuçları ---")
        test_samples = val_inputs[:5] if len(val_inputs) >= 5 else val_inputs
        for i, sample in enumerate(test_samples):
            print(f"Test {i+1}:")
            print(f"Girdi: {sample}")
            response = self.generate_response(sample)
            print(f"Model Çıktısı: {response}")
            print(f"Beklenen: {val_targets[i]}")
            print("-" * 50)

    def generate_response(self, input_text, max_length=512, num_beams=5, temperature=0.8, 
                         do_sample=True, top_p=0.9, repetition_penalty=1.2):
        input_text = normalize_price(normalize_specs(preprocess_text(input_text)))
        formatted_input = f"telefon onerisi: {input_text}"

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
                do_sample=do_sample,
                top_p=top_p,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Tekrarları azalt
                repetition_penalty=repetition_penalty,
                length_penalty=1.1,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def load_trained_model(self, model_path):
        print(f"Model yükleniyor: {model_path}")
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            
            # Data stats'ı yükle
            stats_path = os.path.join(model_path, 'data_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.data_stats = json.load(f)
            
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

def main():
    print("Optimize Edilmiş T5 Telefon Önerisi Model Eğitimi")
    print("=" * 60)
    print("GPU kullanımı:", "Evet" if torch.cuda.is_available() else "Hayır")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    try:
        print("\nModel oluşturuluyor...")
        model = PhoneT5Model('t5-base')  # 40GB VRAM ile t5-base optimal
        model.model.to(device)

        print("Eğitim başlıyor...")
        model.train(
            data_file='training_data.txt',
            epochs=35,  # Daha fazla epoch ama early stopping ile
            batch_size=16,  # 40GB VRAM için optimize edildi
            learning_rate=2e-5  # Büyük batch ile daha yüksek LR
        )

        print("\nEğitim tamamlandı!")
        print("\n" + "="*60)
        print("KAPSAMLI TEST")
        print("="*60)

        test_queries = [
            "15000 tl altinda 6 gb ram android telefon oner",
            "10000 tl 4 gb ram iphone telefon oner", 
            "25000 tl hafizasi iyi olan oyun icin telefon android",
            "ucuz telefon oner 5000 tl",
            "kamerasi iyi telefon 20000 tl samsung"
        ]

        for query in test_queries:
            print(f"\nTest: {query}")
            result = model.generate_response(query)
            print(f"Sonuç: {result}")
            print("-" * 40)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
