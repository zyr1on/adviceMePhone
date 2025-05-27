import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import re

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
    def __init__(self, model_name='t5-base'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.resize_token_embeddings(len(self.tokenizer))

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
                print(f"Satır {line_num}: Geçersiz format (-> bulunamadı): {line[:50]}...")
                continue
            try:
                input_text, output_text = line.split(" -> ", 1)
                input_text = preprocess_text(input_text)
                output_text = preprocess_text(output_text)
                if not input_text or not output_text:
                    print(f"Satır {line_num}: Boş içerik")
                    continue
                inputs.append(input_text)
                targets.append(output_text)
                valid_lines += 1
            except Exception as e:
                print(f"Satır {line_num} işleme hatası: {e}")
                continue
        
        print(f"Toplam {valid_lines} geçerli örnek yüklendi (toplam satır: {len(lines)})")
        if valid_lines > 0:
            print("\n--- Örnek Veriler ---")
            for i in range(min(3, len(inputs))):
                print(f"Girdi: {inputs[i]}")
                print(f"Çıktı: {targets[i]}")
                print("-" * 50)
        
        return inputs, targets
    
    def train(self, data_file, output_dir='./phone_t5_model', epochs=30, batch_size=4, learning_rate=1e-5):
        inputs, targets = self.load_data(data_file)
        if len(inputs) == 0:
            print("HATA: Hiç veri yüklenemedi!")
            return
        
        test_size = 0.15 if len(inputs) >= 20 else 0.1
        
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Eğitim örnekleri: {len(train_inputs)}, Doğrulama örnekleri: {len(val_inputs)}")
        
        train_dataset = PhoneDataset(train_inputs, train_targets, self.tokenizer)
        val_dataset = PhoneDataset(val_inputs, val_targets, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=learning_rate,
            warmup_steps=0,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy='steps',
            eval_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        print("Model eğitimi başlıyor...")
        print(f"Cihaz: {next(self.model.parameters()).device}")
        
        trainer.train()
        
        print("Model kaydediliyor...")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model {output_dir} klasörüne kaydedildi!")
        
        print("\n--- Test Sonuçları ---")
        test_samples = train_inputs[:3] if len(train_inputs) >= 3 else train_inputs
        for i, sample in enumerate(test_samples):
            print(f"Test {i+1}:")
            print(f"Girdi: {sample}")
            response = self.generate_response(sample)
            print(f"Model Çıktısı: {response}")
            print(f"Beklenen: {train_targets[i]}")
            print("-" * 50)
    
    def generate_response(self, input_text, max_length=512, num_beams=1, temperature=0.7):
        input_text = preprocess_text(input_text)
        formatted_input = f"telefon onerisi: {input_text}"
        
        input_ids = self.tokenizer.encode(
            formatted_input,
            return_tensors='pt',
            max_length=256,
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
                do_sample=False,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                length_penalty=1.0
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
    print("T5 Telefon Önerisi Model Eğitimi")
    print("=" * 50)
    print("GPU kullanımı:", "Evet" if torch.cuda.is_available() else "Hayır")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    try:
        print("\nModel oluşturuluyor...")
        model = PhoneT5Model('t5-base')
        model.model.to(device)
        
        print("Eğitim başlıyor...")
        model.train(
            data_file='training_data.txt',
            epochs=10,
            batch_size=8,
            learning_rate=3e-5
        )
        
        print("\nEğitim tamamlandı!")
        print("\n" + "="*50)
        print("MANUEL TEST")
        print("="*50)
        
        test_queries = [
            "15000 tl altinda 6 gb ram android telefon oner",
            "10000 tl 4 gb ram iphone telefon oner",
            "25000 tl hafizasi iyi olan oyun icin telefon android"
        ]
        
        for query in test_queries:
            print(f"\nTest: {query}")
            result = model.generate_response(query)
            print(f"Sonuç: {result}")
            print("-" * 30)
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        print("\nGerekli paketler: pip install torch transformers scikit-learn")

if __name__ == "__main__":
    main()
