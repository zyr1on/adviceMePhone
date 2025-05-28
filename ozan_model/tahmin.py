import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Etiket kategorileri ve karşıtları
conflicting_groups = [
    {"good_camera", "bad_camera"},
    {"apple", "android"},
    {"samsung", "ios"},
    {"ios", "android"},
    # Buraya sen etiketlerine göre istediğin kadar ekleyebilirsin
]

def add_label_with_conflict_check(predicted_labels, new_label):
    for group in conflicting_groups:
        if new_label in group:
            # Aynı gruptaki diğer etiketleri çıkar
            for label in list(predicted_labels):
                if label in group and label != new_label:
                    predicted_labels.remove(label)
    # Yeni etiketi ekle
    if new_label not in predicted_labels:
        predicted_labels.append(new_label)


# ======================== #
# 1. cihaz ayarı (cpu / gpu)
# ======================== # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================== #
# etiketleri yükleyelim (label.txt)
# ======================== #

with open("labels.txt", "r", encoding="utf-8") as f:
    raw = f.read() 

# ======================== #
# - good, bad gibi etiketleri bulmak için regex kullanıyoruz
# ======================== #
import re
labels = re.findall(r"-(.+)", raw) #r regex ile etiketleri bul, "-(.+)" ile - ile başlayan ve sonrasında gelen her şeyi al
labels = sorted(set([l.strip() for l in labels]))  # etiketleri temizle, başındaki ve sonundaki boşlukları kaldır
print(labels)  # Etiketlerin doğru şekilde yüklendiğini kontrol edin

# ======================== #
#  sayisal id eslemeleri :
# ======================== #

id2label = {i: label for i, label in enumerate(labels)}  # id'den etikete
print(id2label)  # Etiketlerin doğru şekilde eşleştirildiğini kontrol edin

# ======================== #
# 3. tokenizer ve egtitilen modeli yükle    
# ======================== #
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
from transformers import DistilBertConfig

# Model yapılandırmasını yeniden tanımla
config = DistilBertConfig(
    num_labels=len(labels),
    problem_type="multi_label_classification"
)

# Yapılandırmaya uygun boş model oluştur
model = DistilBertForSequenceClassification(config)


# ======================== #
# egitilmiş agırlıkları yüklüyoruz
# ======================== #

model.load_state_dict(torch.load("model.pt", map_location=device)) # modelin ağırlıklarını yükle
model.to(device) # modeli cihaza taşıyoruz
model.eval() # Modeli değerlendirme moduna al

# ======================== #
#   kullanıcıdan cümle al ve tahmin et
# ======================== #

cumle = input("Kullanıcıdan cümlesi:")

# TOKENİZE EDİYOZZZ

encoded = tokenizer.encode_plus(
    cumle,
    add_special_tokens=True,  # [CLS] ve [SEP] ekle CLS = başlangıç tokeni, SEP = cümle sonu tokeni
    padding='max_length',  # Maksimum uzunlığa kadar doldur
    truncation=True,  # Çok uzun cümleleri kes
    max_length=32,  # Maksimum token sayısı
    return_tensors="pt"  # PyTorch tensör olarak döndür
)

input_ids = encoded['input_ids'].to(device)  # Token ID'lerini cihaza taşı
attention_mask = encoded["attention_mask"].to(device)  # Padding maskesini cihaza taşı
# padding maskesi, hangi tokenların gerçek olduğunu gösterir (1 = gerçek, 0 = pad)

with torch.no_grad():  # Değerlendirme sırasında gradyan hesaplamaya gerek yok
    #gradyan hesaplama bu işlemde gerekli değil, çünkü model sadece tahmin yapacak
    outputs=model(input_ids=input_ids, attention_mask=attention_mask)  # Modeli çalıştır
    logits = outputs.logits  # Modelin çıktısı (logit değerleri)
    probs = torch.sigmoid(logits)[0]
    print(probs)  # Modelin döndürdüğü olasılık değerlerini kontrol edin

# ======================== #
# belirli eşik değeri (threshold) üstünedeki etiketleri al
# ======================== #
THRESHOLD = 0.41  # Eşik değeri
predicted_labels = [id2label[i] for i, prob in enumerate(probs) if prob > THRESHOLD]

#sonucu yazdır
print("\n tahmin edilen etiketler🧠:")
if predicted_labels: 
    for tag in predicted_labels:
        print(f"- {tag}")
else:
    print("Etiket bulunamadı. Eşik değerinin altında kalan etiketler var.")

# ======================== #
# kullanıcıdan bilgi alma 
# ======================== #

print("\nNe tarz bir telefon arıyorsunuz? (örn: iyi kamera, uzun pil ömrü)")
cumle = input("Lütfen belirtiniz: ")

# TOKENİZE EDİYOZZZ

encoded = tokenizer.encode_plus( #burada kullanıcıdan gelen cümleyi tokenize ediyoruz
    cumle,
    add_special_tokens=True,
    padding="max_length",
    truncation=True, # Çok uzun cümleleri kes
    max_length=32,  # Maksimum token sayısı
    return_tensors="pt"  # PyTorch tensör olarak döndür
)

input_ids = encoded["input_ids"].to(device)  # Token ID'lerini cihaza taşı
attention_mask = encoded["attention_mask"].to(device)  # Padding maskesini cihaza taşı

# tahmin etme kısmı

with torch.no_grad():  # Değerlendirme sırasında gradyan hesaplamaya gerek yok
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Modeli çalıştır
    logits = outputs.logits 
    probs = torch.sigmoid(logits)[0]  # Olasılık değerlerini hesapla

    print("\Olasılıklar(etiket:olasılık):")
    for i, prob in enumerate(probs):
        print(f"{id2label[i]}:{prob.item():.4f}")

# ========================= #
# threshold'a göre tahmin etiketleri: 
# ========================= #

THRESHOLD = 0.41 
predicted_labels = [id2label[i] for i, prob in enumerate(probs) if prob > THRESHOLD]

# ========================= #                            |
# etiketleri kontrol et hiç etiket yoksa soru sorucaz   \|/
# ========================= #                            V
print("\nTahmin edilen etiketler:")

# Eğer tahmin edilen etiket sayısı 3'ten azsa, kullanıcıdan ek bilgi ister
while len(predicted_labels) < 3:
    cumle = input("Yeterli bilgi alınamadı. Daha fazla detay verebilir misiniz?\n→ ")

    # Kullanıcı cümlesini tokenize et
    encoded = tokenizer.encode_plus(
        cumle,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Modeli kullanarak tahmin yap (gradyan hesaplama kapalı)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0]

    # Tahmin edilen etiketleri güncelle
    for i, prob in enumerate(probs):
        if prob > THRESHOLD:
            tag = id2label[i]
            add_label_with_conflict_check(predicted_labels, tag)



    # Güncellenen etiketleri yazdır
    print("\nGüncellenmiş tercihler:")
    for tag in predicted_labels:
        print(f"- {tag}")

# Döngüden çıkınca tüm etiketleri yazdırabiliriz
print("\nSon tahmin edilen etiketler:")
for tag in predicted_labels:
    print(f"- {tag}")

# ========================= #                            ^
# etiketleri kontrol et hiç etiket yoksa soru sorucaz   /|\
# ========================= #                            |


        
    print("\nGüncellenmiş tahmin edilen etiketler:")
    for tag in predicted_labels:
        print(f"- {tag}")

elif len(predicted_labels) < 2: 
    print(f"\nŞuana kadar algılanan özellikler: {predicted_labels}")
    print("Birkaç Özellik daha ekleyebilir misiniz? (örn: iyi kamera, uzun pil ömrü)")
    # burada devam
else:
    print("\nVerdiğiniz bilgiler için teşekkür ederim. Şimdi size uygun telefonları bulmaya çalışacağım.")
    print("\nAlgılanan Telefon özellikleri:")
    for tag in predicted_labels:
        print(f"- {tag}")
