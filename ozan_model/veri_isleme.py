import torch # pytorch : tensör işlemleri, dataset oluşturmak için
from transformers import DistilBertTokenizer # huggingface distilbert tokenizer'i
from torch.utils.data import TensorDataset # TensorDataset, verileri gruplayıp modele vermek için
import os # dosya yolları ve dosya işlemleri için

DATA_PATH = "data.txt" # veri dosyasının yolu
LABEL_PATH = "labels.txt" # etiket dosyasının yolu

# ==== ETİKETLERİ OKU VE TEMİZLE ====
with open(LABEL_PATH, "r", encoding="utf-8") as f: #r dosya okuma modu 
    raw = f.read() # labels.txt dosyasını komple strigng olarak oku

import re #regex ile yazılmış etiketleri bulmak için    

labels = re.findall(r"\- (.+)", raw)    # regex ile etiketleri bul, 
                                        # - ile başlayan ve sonrasında gelen her şeyi al

labels = [label.strip() for label in labels] # etiketleri temizle, başındaki ve sonundaki boşlukları kaldır

labels = sorted(set(labels))    # etiketleri benzersiz yap 
                                #ve sıralı hale getir

label2id = {label: i for i, label in enumerate(labels)}  # etiketleri id'ye çevir

# === Cümleleri ve etiket vektörlerini topla ===

sentences = []       # Her satırdan çıkarılan cümleler burada tutulacak
label_vectors = []   # Her cümleye karşılık gelen etiket vektörü (multi-label)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if "=>" not in line:
            continue  # Eğer satırda => yoksa o satırı atla (boş ya da hatalı olabilir)

        # Cümle ve etiket kısmını ayır (örnek: "bu bir cümle => label1, label2")
        text, label_part = line.strip().split("=>")

        sentence = text.strip()  # Cümle kısmı
        example_labels = [l.strip() for l in label_part.split(",")]  # Etiketleri virgüle göre ayır

        # Vektör başta tamamen 0'larla dolu, kaç etiket varsa o kadar
        vec = [0] * len(label2id)

        # Hangi etiket varsa, onun index'ine 1 yaz (multi-label one-hot encoding)
        for l in example_labels:
            if l in label2id:
                vec[label2id[l]] = 1

        # Verileri listelere ekle
        sentences.append(sentence)
        label_vectors.append(vec)

# === Tokenizer'ı Yükle ===# === Tokenizer hazırla ===
# DistilBERT modeline uygun tokenizer'ı indir (önceden yüklenmişse cache'den alır)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

input_ids = []       # Token ID’leri
attention_masks = [] # Padding maskeleri (1 = gerçek token, 0 = pad)

# Her cümleyi token'lara çevir (DistilBERT formatına)
for sent in sentences:
    encoded = tokenizer.encode_plus(
        sent,                   # Girdi cümlesi
        add_special_tokens=True,   # [CLS], [SEP] gibi özel tokenlar ekle
        padding='max_length',      # En uzun cümle kadar boşlukla doldur
        truncation=True,           # Çok uzun cümleleri kes
        max_length=32,             # Maksimum token sayısı (DistilBERT için ideal)
        return_tensors="pt"        # PyTorch tensör olarak döndür
    )

    input_ids.append(encoded['input_ids'][0])             # Token ID’leri
    attention_masks.append(encoded['attention_mask'][0])  # Maskeleme (pad kontrolü)

# === TensorDataset hazırla ===

# Cümle token'larını, attention maskelerini ve etiketleri tensör olarak birleştir
input_ids = torch.stack(input_ids)                     # [203 x 32]
attention_masks = torch.stack(attention_masks)         # [203 x 32]
labels_tensor = torch.tensor(label_vectors)            # [203 x etiket sayısı] (multi-label)

# Hepsini PyTorch'un TensorDataset objesiyle paketle
dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

# === Eğitime hazır veriyi kaydet ===
torch.save(dataset, "dataset.pt")  # Model eğitiminde kullanılmak üzere dosyaya yaz

print("✅ Veri başarıyla işlendi ve 'dataset.pt' olarak kaydedildi.")

