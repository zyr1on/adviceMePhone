import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
#import pandas as pd
from sklearn.model_selection import train_test_split
#import os

# türkce karakter normalizasyonu
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


class PhoneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = "Telefon önerisi: " + str(self.inputs[idx])
        target_text = str(self.targets[idx])
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }


class PhoneT5Model:
    def __init__(self, model_name='t5-small'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = {
            'additional_special_tokens': ['<phone>', '<feature>', '<value>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    # egitim verilerini yükle, küçük harfe çevir, normalize et.
    def load_data(self, data_file):
        inputs = []
        targets = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        totalline = 0
        for line in lines:
            line = line.strip()
            if not line:  # boş satır kontrolü
                continue
            if line.startswith("//") or " -> " not in line: # // ile başlıyorsa veya -> yoksa discard et
                continue  # yorum satırı veya geçersiz satır atla
            input_text, output_text = line.split(" -> ", 1)
            totalline += 1
            # Küçük harfe çevir
            input_text = input_text.lower()
            output_text = output_text.lower()

            # Türkçe karakterleri çevir
            input_text = turkce_karakterleri_cevir(input_text)
            output_text = turkce_karakterleri_cevir(output_text)

            inputs.append(input_text.strip())
            targets.append(output_text.strip())
        print("Total lines: ", totalline)
        return inputs, targets
    
    def train(self, data_file, output_dir='./phone_t5_model', epochs=3, batch_size=8):
        print("Veri yükleniyor...")
        inputs, targets = self.load_data(data_file)
        
        print(f"Toplam {len(inputs)} örnek yüklendi.")
        
        # Train-validation split
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = PhoneDataset(train_inputs, train_targets, self.tokenizer)
        val_dataset = PhoneDataset(val_inputs, val_targets, self.tokenizer)

        # Training arguments EPOCH
        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     num_train_epochs=epochs,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     learning_rate=3e-5,
        #     warmup_steps=100,
        #     weight_decay=0.01,
        #     logging_dir='./logs',

        #     # CHANGED AREA
        #     evaluation_strategy='epoch',
        #     save_strategy='epoch',
        #     logging_strategy='epoch',


        #     save_total_limit=3,
        #     load_best_model_at_end=True,
        #     metric_for_best_model='eval_loss',
        #     greater_is_better=False,
        #     report_to="none",  # wandb devre dışı
        #     dataloader_num_workers=0  # Windows uyumluluğu
        # )
        
        # Training arguments STEPS
        training_args = TrainingArguments(
            output_dir=output_dir,

            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, # 8 üstü ideal şuan için
            per_device_eval_batch_size=batch_size,  # 8 üstü ideal şuan için
            
            learning_rate=3e-5, 
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            
            eval_strategy='steps',
            eval_steps=50,
            save_steps=100,
            save_total_limit=1,
            
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            
            report_to="none",  # Disable wandb
            dataloader_num_workers=0  # Windows compatibility 
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        print("Model eğitimi başlıyor...")
        trainer.train()
        
        # Save model
        print("Model kaydediliyor...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model {output_dir} klasörüne kaydedildi!")
    
    def generate_response(self, input_text, max_length=256):
        #input_text = "Telefon önerisi: " + input_text
        # telefon önerisi eklenebilir başına
        
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


if __name__ == "__main__":
    print("T5 Model Eğitimi Başlıyor...")
    print("GPU kullanımı:", "Evet" if torch.cuda.is_available() else "Hayır")
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    try:
        # Create model
        model = PhoneT5Model('t5-base')  # t5-base veya t5-large seçilebilir
        model.model.to(device)
        # Train
        model.train('training_data.txt', epochs=15, batch_size=8)  # Batch size düşük tutuldu RAM için        
        print("Eğitim tamamlandı!")
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Gerekli paketler: pip install torch transformers datasets")
