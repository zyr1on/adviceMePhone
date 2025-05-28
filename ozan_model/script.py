import re
from collections import defaultdict

# 1. Etiket listesini yükle
with open("labels.txt", "r", encoding="utf-8") as f:
    raw = f.read()
labels = re.findall(r"- (.+)", raw)
valid_labels = set([l.strip() for l in labels])

# 2. Data dosyasını yükle
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 3. İstatistik tut
etiket_sayaci = defaultdict(int)
gecersiz_etiketler = set()
hatalar = []

# 4. Satır satır işle
for i, line in enumerate(lines):
    if "=>" not in line:
        hatalar.append((i+1, "=> eksik"))
        continue

    try:
        cumle, etiketler_raw = line.strip().split("=>")
        etiketler = [e.strip() for e in etiketler_raw.split(",")]
        for etiket in etiketler:
            if etiket in valid_labels:
                etiket_sayaci[etiket] += 1
            else:
                gecersiz_etiketler.add(etiket)
                hatalar.append((i+1, f"Geçersiz etiket: {etiket}"))
    except:
        hatalar.append((i+1, "Satır ayrıştırılamadı"))

# 5. Sonuçları yazdır
print("\n📊 Etiket Frekansları:")
for etiket in sorted(valid_labels):
    print(f"{etiket:20} : {etiket_sayaci[etiket]}")

if gecersiz_etiketler:
    print("\n🚫 Geçersiz Etiketler:")
    for etiket in gecersiz_etiketler:
        print("-", etiket)

if hatalar:
    print(f"\n⚠️ Hatalı satır sayısı: {len(hatalar)}")
    for satir_no, sorun in hatalar[:10]:
        print(f"Satır {satir_no}: {sorun}")
    if len(hatalar) > 1000:
        print("... ve daha fazlası var.")
else:
    print("\n✅ Hiçbir hata bulunamadı. Veri dosyan temiz.")
