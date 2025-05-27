import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import re
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def turkce_karakterleri_cevir(text):
    cevirme_dict = {
        'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U'
    }
    for tr_char, en_char in cevirme_dict.items():
        text = text.replace(tr_char, en_char)
    return text

def advanced_preprocess_text(text):
    text = text.lower().strip()
    text = turkce_karakterleri_cevir(text)
    # Sayılar ve önemli karakterleri koru
    text = re.sub(r'\s+', ' ', text)
    # Sadece gereksiz noktalama işaretlerini kaldır, : ve ; gibi önemli olanları koru
    text = re.sub(r'[^\w\s:;-]', '', text)
    return text

class OptimizedPhoneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_length=128, max_target_length=256):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Veri augmentation için
        self.augment_data()
    
    def augment_data(self):
        """Veri çoğaltma - eş anlamlı kelimeler ve varyasyonlar"""
        original_inputs = self.inputs.copy()
        original_targets = self.targets.copy()
        
        synonyms = {
            'telefon': ['telefon', 'cep telefonu', 'mobil'],
            'oner': ['oner', 'onerir misin', 'tavsiye et'],
            'iyi': ['iyi', 'kaliteli', 'guzel'],
            'ucuz': ['ucuz', 'uygun fiyatli', 'ekonomik'],
            'android': ['android', 'android isletim sistemi'],
            'iphone': ['iphone', 'apple telefon', 'ios telefon']
        }
        
        # Her örnek için 1-2 varyasyon oluştur
        for i in range(len(original_inputs)):
            if len(self.inputs) < len(original_inputs) * 3:  # Max 3x artır
                augmented = original_inputs[i]
                for word, variants in synonyms.items():
                    if word in augmented and np.random.random() > 0.7:
                        new_word = np.random.choice(variants)
                        augmented = augmented.replace(word, new_word, 1)
                
                if augmented != original_inputs[i]:
                    self.inputs.append(augmented)
                    self.targets.append(original_targets[i])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = f"translate: {str(self.inputs[idx])}"
        target_text = str(self.targets[idx])
        
        # Input tokenization
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        # Target tokenization - T5 için önemli
        with self.tokenizer.as_target_tokenizer():
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

class OptimizedPhoneT5Model:
    def __init__(self, model_name='google/flan-t5-base'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Özel token'lar ekle
        special_tokens = ['<price>', '<brand>', '<os>', '<usage>', '<ram>', '<storage>', '<battery>', '<camera>', '<screen>']
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Gradual unfreezing için katman sayılarını belirle
        self.setup_gradual_unfreezing()

    def setup_gradual_unfreezing(self):
        """Progressif unfreezing için hazırlık"""
        # İlk başta sadece decoder'ı eğit
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze_layers(self, stage):
        """Aşamalı olarak katmanları serbest bırak"""
        if stage == 1:  # Encoder'ın son katmanlarını aç
            for name, param in self.model.named_parameters():
                if "encoder.block.1" in name or "encoder.block.2" in name:
                    param.requires_grad = True
        elif stage == 2:  # Tüm encoder'ı aç
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True

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
            if not line or line.startswith("//"):
                continue
            if " -> " not in line:
                continue
            
            try:
                input_text, output_text = line.split(" -> ", 1)
                input_text = advanced_preprocess_text(input_text)
                # Output'u çok fazla işleme, yapısını koru
                output_text = output_text.strip()
                
                if not input_text or not output_text:
                    continue
                    
                inputs.append(input_text)
                targets.append(output_text)
                valid_lines += 1
            except Exception as e:
                continue
        
        print(f"Toplam {valid_lines} geçerli örnek yüklendi")
        return inputs, targets
    
    def train_progressive(self, data_file, output_dir='./optimized_phone_t5', total_epochs=50):
        inputs, targets = self.load_data(data_file)
        if len(inputs) == 0:
            print("HATA: Hiç veri yüklenemedi!")
            return
        
        # Veri bölme - validation için daha az ayır
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.1, random_state=42, shuffle=True
        )
        
        print(f"Eğitim: {len(train_inputs)}, Doğrulama: {len(val_inputs)}")
        
        # 3 aşamalı eğitim - daha conservative
        stages = [
            {"epochs": 10, "lr": 1e-4, "batch_size": 4, "stage": 0},
            {"epochs": 10, "lr": 5e-5, "batch_size": 4, "stage": 1}, 
            {"epochs": 10, "lr": 1e-5, "batch_size": 4, "stage": 2}
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for stage_idx, stage_config in enumerate(stages):
            print(f"\n=== STAGE {stage_idx + 1}/{len(stages)} ===")
            print(f"Epochs: {stage_config['epochs']}, LR: {stage_config['lr']}")
            
            # Katman dondurmayı güncelle
            self.unfreeze_layers(stage_config['stage'])
            
            # Dataset oluştur
            train_dataset = OptimizedPhoneDataset(train_inputs, train_targets, self.tokenizer)
            val_dataset = OptimizedPhoneDataset(val_inputs, val_targets, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/stage_{stage_idx}",
                num_train_epochs=stage_config['epochs'],
                per_device_train_batch_size=stage_config['batch_size'],
                per_device_eval_batch_size=stage_config['batch_size'],
                learning_rate=stage_config['lr'],
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_steps=5,
                eval_strategy='steps',
                eval_steps=25,
                save_steps=50,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                report_to="none",
                gradient_accumulation_steps=1,
                fp16=False,  # FP16 kapatıldı - NaN sebep olabilir
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                prediction_loss_only=False,  # Loss hesabı için
                gradient_checkpointing=False,  # İlk başta kapalı
                max_grad_norm=1.0,  # Gradient clipping
                adam_epsilon=1e-8,
                lr_scheduler_type="linear",
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
            )
            
            print(f"Stage {stage_idx + 1} eğitimi başlıyor...")
            trainer.train()
            
            # Her stage sonunda test
            print(f"\n--- Stage {stage_idx + 1} Test ---")
            test_samples = train_inputs[:2]
            for i, sample in enumerate(test_samples):
                response = self.generate_response(sample)
                print(f"Test: {sample}")
                print(f"Çıktı: {response}")
                print(f"Beklenen: {train_targets[i]}")
                print("-" * 40)
        
        # Final model save
        print("Final model kaydediliyor...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model {output_dir} klasörüne kaydedildi!")

    def generate_response(self, input_text, max_length=256, num_beams=5, temperature=0.7):
        input_text = advanced_preprocess_text(input_text)
        formatted_input = f"translate: {input_text}"
        
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
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def load_trained_model(self, model_path):
        print(f"Model yükleniyor: {model_path}")
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

def main():
    print("Optimized T5 Telefon Önerisi Model Eğitimi")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU: {'Evet' if torch.cuda.is_available() else 'Hayır'}")
    print(f"Cihaz: {device}")
    
    try:
        print("\nOptimized model oluşturuluyor...")
        model = OptimizedPhoneT5Model('google/flan-t5-base')
        
        print("Progressive eğitim başlıyor...")
        model.train_progressive(
            data_file='training_data.txt',
            total_epochs=50
        )
        
        print("\n" + "="*50)
        print("FINAL TEST")
        print("="*50)
        
        test_queries = [
            "oyun oynamak icin android telefon oner",
            "kamerasi iyi olan samsung telefon",
            "15000 tl altinda 6 gb ram telefon",
            "fotograf cekmek icin iphone"
        ]
        
        for query in test_queries:
            print(f"\nTest: {query}")
            result = model.generate_response(query)
            print(f"Sonuç: {result}")
            print("-" * 30)
        
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
