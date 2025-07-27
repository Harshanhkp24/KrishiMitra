from flask import Flask, request, jsonify, render_template
import joblib
import csv
from datetime import datetime
import os


app = Flask(__name__)

# Load the trained model and label map
model = joblib.load("crop_model.pkl")
label_map = joblib.load("label_map.pkl")

# Basic crop tips
crop_tips = {
    "Wheat": "Use well-drained loamy soil and irrigate moderately.",
    "Rice": "Ensure standing water in early growth stages.",
    "Maize": "Needs full sunlight and regular watering.",
    "Sugarcane": "Requires heavy rainfall or good irrigation.",
    "Cotton": "Best grown in black soil and warm climates.",
    "Groundnut": "Prefers sandy loam soil with proper drainage.",
    "Pulses": "Use phosphorus-rich fertilizer for better yield.",
    "Millets": "Drought-tolerant and ideal for less fertile soil.",
    "Mustard": "Use well-prepared seedbeds and proper spacing.",
    "Barley": "Cool climate crop with moderate water needs."
}


# Soil type encoding (same as used during training)
soil_encoding = {
    "Loamy": 1,
    "Sandy": 2,
    "Clay": 0
}

@app.route("/")
def index():
    return render_template("index.html")

# New Route: Render HTML Form
@app.route("/form")
def crop_form():
    return render_template("form.html")

# New Route: Handle HTML form POST
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        soil_type = request.form["soil_type"]
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])

        if soil_type not in soil_encoding:
            return render_template("index.html", prediction=f"Invalid soil_type '{soil_type}'")

        soil_code = soil_encoding[soil_type]
        input_features = [[soil_code, rainfall, temperature]]
        prediction_code = model.predict(input_features)[0]
        predicted_crop = label_map[prediction_code]
        tip = crop_tips.get(predicted_crop, "No tip available for this crop.")

         # Save prediction to history.csv
        
        history_file = "prediction_history.csv"
        file_exists = os.path.isfile(history_file)

        with open(history_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "soil_type", "rainfall", "temperature", "predicted_crop"])
            writer.writerow([datetime.now(), soil_type, rainfall, temperature, predicted_crop])



        return render_template("form.html", prediction=predicted_crop, tip=tip)
            
       
    


    except Exception as e:
        return render_template("form.html", prediction=f"Error: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

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

        # For web form, show result as HTML
        if not request.is_json:
            return f"<h2>Predicted Crop: {predicted_crop}</h2>"

        return jsonify({
            "soil_type": soil_type,
            "rainfall": rainfall,
            "temperature": temperature,
            "predicted_crop": predicted_crop
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/history")
def history():
    import csv

    history_data = []
    try:
        with open("prediction_history.csv", mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                history_data.append(row)
    except FileNotFoundError:
        history_data = []

    # Convert to chart-friendly format
    predictions = {
        "labels": [],
        "rainfall": [],
        "temperature": [],
        "crops": []
    }

    for entry in history_data:
        predictions["labels"].append(entry["timestamp"] if "timestamp" in entry else len(predictions["labels"]))
        predictions["rainfall"].append(float(entry["rainfall"]))
        predictions["temperature"].append(float(entry["temperature"]))
        predictions["crops"].append(entry["prediction"])

    return render_template("history.html", history=history_data, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
