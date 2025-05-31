try:
    from flask import Flask, render_template, request
    from phone_recommender import PhonePredictor
    from phone_recommender2 import PhonePredictor2
    import os
except ImportError as e:
    print("\n[!] Gerekli modüller yüklenemedi:")
    print(f"    -> {e}")
    print("\nLütfen gerekli bağımlılıkları yüklemek için aşağıdaki komutu çalıştırın:")
    print("    python3 install.py\n")

app = Flask(__name__)

# Model seçimi için başlatıcı
def get_predictor(model_choice):
    """Model seçimine göre doğru tahmin edici nesnesini döndürür."""
    if model_choice == "pkl":
        return PhonePredictor(model_path="enhanced_phone_model.pkl", data_path="phones.csv")
    elif model_choice == "pt":
        if not os.path.exists("model.pt"):
            raise FileNotFoundError("model.pt bulunamadı. Lütfen 'python.exe install.py' komutunu çalıştırın.")
        return PhonePredictor2(model_path="model.pt", data_path="phones.csv", labels_path="labels.txt")
    else:
        raise ValueError("Geçersiz model seçimi!")

@app.route("/", methods=["GET"])
def index():
    """Ana sayfa"""
    return render_template("index.html")

@app.route("/select_model", methods=["POST"])
def select_model():
    """Model seçimi"""
    model_choice = request.form.get("model_choice", "pkl")
    return render_template("index.html", model_choice=model_choice)

@app.route("/recommend", methods=["POST"])
def recommend():
    """Telefon önerisi yap"""
    try:
        user_input = request.form.get("prompt", "").strip()
        sort_order = request.form.get("sort_order", "asc")
        model_choice = request.form.get("model_choice", "pkl")

        if not user_input:
            return render_template("index.html", error="Lütfen bir şeyler yazın!", prompt=user_input, model_choice=model_choice)

        # Doğru predictor'ı başlat
        predictor = get_predictor(model_choice)

        # Tahmin yap
        predictions_list, confidences_list, raw_outputs, extracted_values_list = predictor.predict(user_input)

        # Önerileri al
        recommendations = predictor.recommend_phones(predictions_list, confidences_list, extracted_values_list)

        # Çıktıyı formatla
        recommendations_list = []
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                phone_info = {
                    "brand": row["brand"],
                    "price": f"{row['price']:.0f} TL",
                    "price_numeric": row['price'],
                    "os": row["os"],
                    "storage": row["storage"],
                    "ram": f"{row['ram']:.0f}GB",
                    "camera": row["camera"],
                    "battery": row["battery"],
                    "screen": row["screen"],
                    "usage": row["usage"],
                    "link": row.get("link", "#")
                }
                recommendations_list.append(phone_info)

        # Fiyata göre sırala
        recommendations_list.sort(key=lambda x: x["price_numeric"], reverse=(sort_order == "desc"))

        # Uygulanan filtreler
        applied_filters = []
        for predictions, extracted_values in zip(predictions_list, extracted_values_list):
            for feature, value in predictions.items():
                if value != "none" and feature in predictor.phones_df.columns and feature not in ["price", "ram"]:
                    applied_filters.append(f"{feature}={value}")
            if "ram" in extracted_values:
                applied_filters.append(f"ram={extracted_values['ram']}GB")
            if "price" in extracted_values:
                applied_filters.append(f"price<={extracted_values['price']}TL")

        # Ortalama güven skoru
        avg_confidence = sum(sum(c.values()) / len(c) for c in confidences_list) / len(confidences_list) if confidences_list else 0

        return render_template(
            "index.html",
            recommendations=recommendations_list,
            applied_filters=applied_filters,
            avg_confidence=avg_confidence,
            prompt=user_input,
            raw_outputs=raw_outputs,
            extracted_values_list=extracted_values_list,
            sort_order=sort_order,
            model_choice=model_choice
        )

    except Exception as e:
        # Hata durumunda, detaylı hata mesajı veriyoruz
        return render_template("index.html", error=f"Hata oluştu: {str(e)}", prompt=user_input, model_choice=model_choice)

if __name__ == "__main__":
    app.run(debug=True)
