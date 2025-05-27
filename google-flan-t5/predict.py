import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

def turkce_karakterleri_cevir(text):
    cevirme_dict = {
        'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U'
    }
    for tr_char, en_char in cevirme_dict.items():
        text = text.replace(tr_char, en_char)
    return text

def preprocess_text(text):
    text = text.lower().strip()
    text = turkce_karakterleri_cevir(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s:;-]', '', text)
    return text

class PhonePredictor:
    def __init__(self, model_path='./optimized_phone_t5'):
        print("Model yükleniyor...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            print(f"Model yüklendi! Cihaz: {device}")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            print("Model dosyalarının yolunu kontrol edin.")
    
    def predict(self, input_text):
        input_text = preprocess_text(input_text)
        formatted_input = f"query: {input_text}"
        
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
                max_length=256,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

def run_tests():
    predictor = PhonePredictor()
    
    test_cases = [
        "oyun oynamak icin android telefon oner",
        "kamerasi iyi olan samsung telefon",
        "15000 tl altinda android telefon",
        "fotograf cekmek icin iphone telefon",
        "8 gb ram olan android telefon oner"
    ]
    
    print("\n" + "="*60)
    print("TEST SONUÇLARI")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Girdi: {test}")
        result = predictor.predict(test)
        print(f"Çıktı: {result}")
        print("-" * 50)

def interactive_mode():
    predictor = PhonePredictor()
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Telefon önerisi isteyin (çıkmak için 'q' yazın)")
    
    while True:
        user_input = input("\nTelefon isteği: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit', 'çık']:
            print("Çıkılıyor...")
            break
        
        if not user_input:
            print("Lütfen bir şey yazın.")
            continue
        
        try:
            result = predictor.predict(user_input)
            print(f"Sonuç: {result}")
        except Exception as e:
            print(f"Hata: {e}")

def main():
    print("Phone Model Predictor")
    print("=" * 30)
    
    while True:
        print("\nSeçenekler:")
        print("1. Test çalıştır (5 örnek)")
        print("2. Interactive mode")
        print("3. Çıkış")
        
        choice = input("\nSeçiminiz (1-3): ").strip()
        
        if choice == '1':
            run_tests()
        elif choice == '2':
            interactive_mode()
        elif choice == '3':
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main()
