from flask import Flask, render_template, request
from phone_recommender import PhonePredictor

app = Flask(__name__)

# PhonePredictor'ı başlat
predictor = PhonePredictor(model_path="enhanced_phone_model.pkl", data_path="phones.csv")

@app.route("/", methods=["GET"])
def index():
    """Ana sayfa"""
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    """Telefon önerisi yap"""
    try:
        user_input = request.form.get("prompt", "").strip()
        if not user_input:
            return render_template("index.html", error="Lütfen bir şeyler yazın!", prompt=user_input)

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
                    "os": row["os"],
                    "storage": row["storage"],
                    "ram": f"{row['ram']:.0f}GB",
                    "camera": row["camera"],
                    "battery": row["battery"],
                    "screen": row["screen"],
                    "usage": row["usage"]
                }
                recommendations_list.append(phone_info)

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
            extracted_values_list=extracted_values_list
        )

    except Exception as e:
        return render_template("index.html", error=f"Hata oluştu: {str(e)}", prompt=user_input)

if __name__ == "__main__":
    app.run(debug=True)