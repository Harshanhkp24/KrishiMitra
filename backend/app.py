from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and label map
model = joblib.load("crop_model.pkl")
label_map = joblib.load("label_map.pkl")

# Soil type encoding (same as used during training)
soil_encoding = {
    "Loamy": 1,
    "Sandy": 2,
    "Clay": 0
}

@app.route("/")
def index():
    return "🌱 Welcome to KrishiMitra Crop Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        soil_type = data["soil_type"]
        rainfall = float(data["rainfall"])
        temperature = float(data["temperature"])

        if soil_type not in soil_encoding:
            return jsonify({"error": f"Invalid soil_type '{soil_type}'"}), 400

        # Encode soil type
        soil_code = soil_encoding[soil_type]

        # Make prediction
        input_features = [[soil_code, rainfall, temperature]]
        prediction_code = model.predict(input_features)[0]
        predicted_crop = label_map[prediction_code]

        return jsonify({
            "soil_type": soil_type,
            "rainfall": rainfall,
            "temperature": temperature,
            "predicted_crop": predicted_crop
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        soil_type = request.form["soil_type"]
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])

        if soil_type not in soil_encoding:
            return f"❌ Invalid soil type: {soil_type}"

        soil_code = soil_encoding[soil_type]
        input_features = [[soil_code, rainfall, temperature]]

        prediction_code = model.predict(input_features)[0]
        predicted_crop = label_map[prediction_code]

        return render_template("form.html", prediction=predicted_crop)

    except Exception as e:
        return f"Something went wrong: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
