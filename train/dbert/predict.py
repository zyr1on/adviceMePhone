import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import re

# Etiket kategorileri ve karşıtları
conflicting_groups = [
    {"good_camera", "bad_camera"},
    {"apple", "android"},
    {"samsung", "ios"},
    {"ios", "android"},
]

def setup_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Etiketleri yükle
    with open("labels.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    
    labels = re.findall(r"-(.+)", raw)
    labels = sorted(set([l.strip() for l in labels]))
    id2label = {i: label for i, label in enumerate(labels)}

    # Model ve tokenizer'ı ayarla
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    config = DistilBertConfig(
        num_labels=len(labels),
        problem_type="multi_label_classification"
    )
    model = DistilBertForSequenceClassification(config)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device, id2label

def get_predictions(text, model, tokenizer, device, threshold=0.41):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)[0]
    
    return probs

def add_label_with_conflict_check(predicted_labels, new_label):
    for group in conflicting_groups:
        if new_label in group:
            for label in list(predicted_labels):
                if label in group and label != new_label:
                    predicted_labels.remove(label)
    if new_label not in predicted_labels:
        predicted_labels.append(new_label)

def get_missing_categories(predicted_labels):
    # Önemli kategorileri tanımla
    essential_categories = {
        'brand': ['apple', 'samsung', 'android', 'ios'],
        'camera': ['good_camera', 'bad_camera'],
        'battery': ['good_battery', 'bad_battery'],
        'usage': ['gaming', 'business', 'casual']
    }
    
    missing = []
    for category, labels in essential_categories.items():
        if not any(label in predicted_labels for label in labels):
            missing.append(category)
    
    return missing

def main():
    model, tokenizer, device, id2label = setup_model_and_tokenizer()
    predicted_labels = []
    
    print("\nNe tarz bir telefon arıyorsunuz?")
    while True:
        user_input = input("→ ")
        
        probs = get_predictions(user_input, model, tokenizer, device)
        
        # Yeni etiketleri ekle
        for i, prob in enumerate(probs):
            if prob > 0.41:
                add_label_with_conflict_check(predicted_labels, id2label[i])
        
        # Mevcut etiketleri göster
        print("\nŞu ana kadar belirlenen özellikler:")
        for tag in predicted_labels:
            print(f"- {tag}")
            
        # Eksik kategorileri kontrol et
        missing_categories = get_missing_categories(predicted_labels)
        
        if not missing_categories:
            print("\nYeterli bilgi toplandı! İşte telefon tercihiniz için belirlenen özellikler:")
            for tag in predicted_labels:
                print(f"- {tag}")
            break
        else:
            print("\nBiraz daha bilgiye ihtiyacımız var. Lütfen şunlar hakkında bilgi verin:")
            for category in missing_categories:
                if category == 'brand':
                    print("- Tercih ettiğiniz marka veya işletim sistemi?")
                elif category == 'camera':
                    print("- Kamera kalitesi sizin için önemli mi?")
                elif category == 'battery':
                    print("- Pil ömrü beklentiniz nedir?")
                elif category == 'usage':
                    print("- Telefonu hangi amaçla kullanacaksınız?")

if __name__ == "__main__":
    main()



