from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("covid_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # Input JSON
        features = np.array(data["features"]).reshape(1, -1)

        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0][0]

        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
