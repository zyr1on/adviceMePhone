import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

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
    return text


def predict(model, tokenizer, query):
    processed_query = preprocess_text(query)
    formatted_input = f"telefon onerisi: {processed_query}"
    
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt', max_length=128, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Model yükle
model_path = './phone_t5_model'
print("Model yükleniyor...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
print("Model yüklendi!")

# While döngüsü
while True:
    query = input("\nTelefon isteğiniz: ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    result = predict(model, tokenizer, query)
    print(f"Öneri: {result}")
