import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW


import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim cihazı: {device}")

# Dosyayı yükle
dataset = torch.load("dataset.pt", weights_only=False)


batch_size = 8 
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_labels = dataset.tensors[2].shape[1]  # Etiket sayısı
print(f"Etiket sayısı: {num_labels}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,  # Multi-label için etiket sayısı
    problem_type="multi_label_classification"  # Multi-label sınıflandırma
)

model.to(device)  # Modeli CUDA'ya taşı

# === Kayıp fonksiyonu ve optimizer ===
criterion = nn.BCEWithLogitsLoss()  # Çoklu etiket için özel loss
optimizer = AdamW(model.parameters(), lr=2e-5)  # BERT için önerilen öğrenme oranı


# === Eğitim döngüsü ===
epochs = 5

for epoch in range(epochs):
    print(f"\n🔁 Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits  # Modelin tahmini (çıktısı)
        loss = criterion(logits, labels.float())  # Kayıp değeri

        optimizer.zero_grad()  # Önceki gradyanları sıfırla
        loss.backward()        # Geri yayılım
        optimizer.step()       # Ağırlıkları güncelle

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📉 Ortalama epoch kaybı: {avg_loss:.4f}")

# === Modeli kaydet ===
torch.save(model.state_dict(), "model.pt")
print("✅ Model başarıyla eğitildi ve 'model.pt' olarak kaydedildi.")