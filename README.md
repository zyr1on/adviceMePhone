# 📱 Akıllı Telefon Öneri Sistemi (T5-Base + Pandas + FLASK-UI)

Bu proje, kullanıcıdan **doğal dilde** gelen bir prompt'u alarak, bu prompt'u önceden eğitilmiş olarak anlamlandıran bir **T5-Base modelini** kullanır. Model, prompt'tan filtreleme kriterleri çıkartır ve ardından **Pandas** ile hazırlanmış telefon veri seti üzerinde bu kriterlere göre filtreleme yaparak en uygun telefonları önerir. Tüm süreç şık bir **kullanıcı arayüzü** üzerinden yürütülür.

---

### Eğitim Kısmı

Model, Google Colab ortamında NVIDIA A100 GPU kullanılarak toplam **input-output** olacak şekilde 550'ye yakın veri ile birlikte--<br> **50 epoch ve 3e-5 lr** kullanılarak eğitildi.
Eğitim için öğrenme oranını arttırmak için preprocess uygulandı. Bu preprocess işlemi türkçe karakter normalizasyonu ve tolower() kullanılarak yapıldı.

---

## 🧠 Özellikler

- 🔍 doğal dil girdisinden anlamlı filtreler üretir.  
- 🧪 T5-Base ile prompt-to-filter dönüşümü.  
- 📊 pandas ile veri üzerinde hızlı filtreleme.  
- 🎨 sade ve modern kullanıcı arayüzü.  
---

## 🚀 BAŞLANGIÇ

### 1. Depoyu klonlamak için

```bash
git clone https://github.com/zyr1on/adviceMePhone/.git
cd adviceMePhone
```

### 2. Gereksinimleri Kurun

Python 3.8+ sürümü önerilir.
```bash
pip install -r requirements.txt
```

### 3. Eğitim (Opsiyonel)

Modeli yeniden eğitmek istersen:

```bash
python t5/train_preprocess.py
```

> Eğitim dosyası: `t5/datas/training_data.txt`

### 4. Tahmin/Filtreleme Modunu Başlat

```bash
python3 t5/predict.py
```

Arayüz üzerinden yazacağınız prompt örnekleri:

```
"6 gb üzeri android telefon öner"
"oyun için en iyi bataryalı telefon"
```

---

## 📁 Proje Yapısı

```
📦 adviceMePhone
├── 
│   └── 
├── model/
│   ├── 
│   └── 
├── ui/
│   └── 
├── 
├── 
├── 
├── 
└── 
```

---

## 🔧 Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|----------|----------|
| 🧠 [T5-Base](https://huggingface.co/t5-base) | Prompt'tan filtre çıkarma |
| 🐼 Pandas | Veri filtreleme |
| 🖥️ Streamlit / Flask / Custom UI | GPT tarzı kullanıcı arayüzü |
| 💾 CSV | Telefon veri seti (özellikler: RAM, fiyat, marka, batarya vs.) |

---

## 💬 Örnek Promptlar ve Çıktılar

### Prompt:
```
"6000 tl altı 6gb ramli oyun telefonu"
```

### Model Output:
```
price: 6000; brand:none; os:android; ram:6; usage:gaming; storage:none; battery:none; camera:none; screen:none;
```

### Pandas Filtre Sonucu:
| Prie (TL) | Brand  | Model          | İşletim Sistemi | Kullanım Amacı | RAM   | Hafıza | Batarya   | Kamera | Ekran |
|------------|--------|----------------|------------------|----------------|-------|--------|-----------|--------|--------|
| 5800       | Xiaomi | Redmi Note 10  | Android          | Game         | 6 GB  | good | good  | medium  | medium |

.......
---

## ✨ Ekran Görüntüsü


---

## 🧩 Geliştirici Notları

- Eğitim verisi `prompt -> filtre` eşleşmeleri içerir.
- Model eğitimi HuggingFace Transformers ile yapılmıştır.
- Arayüz kısmı, ilk prompt sonrası animasyonlu olarak aşağı kayan sabit prompt kutusu ve typewriter efektiyle çıktı göstermektedir.

---


test

