import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("dosya_adi.csv")

# RAM sütunundaki sayısal kısmı ayıkla ve karşılaştır
df['ram_numeric'] = df['ram'].str.extract(r'(\d+)').astype(float)

# 12 ve üzeri olanlarda usage sütununu 'game' olarak güncelle
df.loc[df['ram_numeric'] >= 12, 'usage'] = 'game'

# İstersen geçici sütunu silebilirsin
df.drop(columns=['ram_numeric'], inplace=True)

# Dosyayı kaydet
df.to_csv("duzenlenmis_dosya.csv", index=False)
