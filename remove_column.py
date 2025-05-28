import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("dosya.csv")

# İlk sütunu kaldır
df = df.iloc[:, 1:]

# Değişiklikleri kaydet (isteğe bağlı olarak yeni bir dosya olarak kaydedebilirsin)
df.to_csv("duzenlenmis_dosya.csv", index=False)
