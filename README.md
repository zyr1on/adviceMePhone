# 📱 Akıllı Telefon Tavsiye Sistemi (BERT, T5, PKL)

Bu proje, doğal dil girdilerine göre en uygun telefonları önermek için eğitilmiş bir yapay zeka sistemidir. Kullanıcıdan alınan doğal dildeki istekler (örneğin: _"oyun için 10 bin altı telefon öner"_) analiz edilerek filtre kriterlerine dönüştürülür ve uygun telefonlar listelenir.

## 👥 Katkıda Bulunanlar

- Semih Özdemir 	(model eğitimi, eğitim veri seti hazırlama )
- Ozan Aydın 		  (model eğitimi, eğitim veri seti hazırlama )
- Eren Boylu 		  (backend, veri temizliği, filtreleme, telefon veriseti )
- Muhsin Yılmaz 	(backend, frontend, eğitim veri seti hazırlama ve düzenleme )


## 🚀 Özellikler

- ✅ Doğal dilden telefon filtreleme (örn: "oyun için 8 GB RAM’li telefonlar")
- 🤖 3 farklı model desteği:
  - **DistilBERT** tabanlı çok etiketli sınıflandırma (`model.pt`)
  - **T5** tabanlı metinden etiket çıkarımı
  - **Traditional ML (.pkl)** modeli (`enhanced_phone_model.pkl`)
- 🌐 Web arayüzü (HTML + JS + CSS)
- 📊 `phones.csv`: telefon veritabanı (özellikleriyle birlikte)

---

# 📁 Proje Yapısı

```
📁 Proje Kök Dizini
├── app.py                      # Flask uygulaması (ana giriş noktası)
├── model.pt                    # BERT tabanlı PyTorch modeli
├── enhanced_phone_model.pkl    # Scikit-learn ile eğitilmiş PKL modeli
├── phones.csv                  # Telefon veri seti
├── labels.txt                  # Etiket tanımları
├── requirements.txt            # Gerekli Python paketleri
├── templates/
│   └── index.html              # Web arayüzü HTML dosyası
├── static/
│   ├── css/
│   │   └── style.css           # CSS stilleri
│   └── js/
│       └── script.js           # JavaScript mantığı
├── train/
│   ├── t5/
│   │   ├── train.py            # T5 model eğitimi
│   │   └── predict.py          # T5 model tahmini
│   └── pkl/
│       ├── train.py            # PKL model eğitimi
│       └── predict.py          # PKL model tahmini
```

## 🧠 Modellerin Açıklaması

### 🔹 1. BERT (PyTorch - `model.pt`)
- Çoklu etiket sınıflandırması.
- Girdi: doğal dil prompt
- Çıktı: `{ "os": "android", "ram": "8", "usage": "game", ... }`

### 🔹 2. T5 (HuggingFace - `train/t5/`)
- Sequence-to-sequence olarak çalışır.
- `train.py`: modeli eğitir.
- `predict.py`: metin girdisini JSON formatlı filtrelere çevirir.

### 🔹 3. PKL (Traditional ML - `train/pkl/`)
- TF-IDF + KNN / Logistic Regression tarzı klasik model.
- Hızlıdır, ancak karmaşık promptları çözmede sınırlıdır.

---

## 🧠 Modellerin Detayları

### 🔹 1. BERT (PyTorch - `model.pt`)

- DistilBERT mimarisi kullanılarak çoklu etiket sınıflandırması yapar.  
- Girdi olarak doğal dilde kullanıcı promptu alır.  
- Çıktı olarak JSON formatında telefon filtreleme kriterleri üretir.  
- Model, derin öğrenme sayesinde karmaşık ve özgün ifadeleri anlayabilir.

---

### 🔹 2. T5 (HuggingFace - `train/t5/`)

- Sequence-to-sequence (metinden metne) modeli.  
- `train.py` ile sıfırdan veya transfer öğrenme ile model eğitilebilir.  
- `predict.py` metin girdisini filtre kriterlerine dönüştürür.  
- Daha esnek ve farklı yapılı promptlar için uygundur.

---

### 🔹 3. PKL (Geleneksel Makine Öğrenimi - `train/pkl/`)

- TF-IDF + KNN veya Logistic Regression gibi klasik yöntemler kullanır.  
- Daha hızlı tahmin yapar ancak karmaşık dil ifadelerinde sınırlı kalabilir.  
- Eğitim ve tahmin scriptleri `train/pkl/` klasöründe bulunur.

---


## ⚙️ Kurulum

```bash
# direkt olarak install.py çalıştırılabilir

python3 install.py

# enhanced_phone_model.pkl dosyası hazır gelmekte
# install.py ile gereklü kütüphaneler ve model dosyaları yüklenebilir.
```

## 🚀 Kullanım

### 1. Sunucuyu başlat

```bash
python app.py
```

Tarayıcıda `http://localhost:5000` adresine gidin.

Prompt kutusuna doğal dilde bir istek girin:

> "10 bin tl altı oyun telefonu önerir misin?"

### 2. Eğitim (isteğe bağlı)

🔹 **T5 modeli eğitimi:**
```bash
cd train/t5
python train.py
```

🔹 **PKL modeli eğitimi:**
```bash
cd train/pkl
python train.py
```

## 📊 Dataset Özellikleri (phones.csv)

Aşağıdaki özellikler modele input olarak verilir:

- **os** (android / ios / none)
- **ram** (GB)
- **price** (TL)
- **storage** (GB)
- **camera** (MP)
- **battery** (mAh)
- **screen** (inç)
- **usage** (oyun / sosyal / kameralı / none)
- **brand** (samsung / xiaomi / apple / ...)
- **link** (ürün linki)

## 📦 API Endpoint (Geliştiriciler için)

### POST /predict

**İstek:**
```json
{
  "input_text": "8 gb ramli android telefon öner"
}
```

**Yanıt:**
```json
{
  "predictions": [
    {
      "os": "android",
      "ram": "8"
    }
  ],
  "confidences": [
    {
      "os:android": 0.91,
      "ram:8": 0.87
    }
  ],
  "text_outputs": [
    "8 gb ramli android telefon öner: os:android; ram:8"
  ]
}
```


