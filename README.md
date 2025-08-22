🧠 COVID-19 Case Predictor (Flask + TensorFlow + Joblib)

This project deploys a deep learning regression model trained to predict daily new COVID-19 cases.
It uses a Keras Sequential model, scales input features with StandardScaler, and serves predictions via a Flask REST API.

📂 Project Structure
```bash
├── app.py               # Flask app serving predictions
├── model.h5             # Trained Keras model (saved using model.save)
├── scaler.pkl           # Pre-fitted StandardScaler (saved with joblib)
├── README.md            # Documentation
├── requirements.txt     # Python dependencies
└── data/                # (Optional) Source dataset used for training
```


⚙️ Installation

Clone this repo:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

📦 Dependencies

Key libraries used:

Flask → Web framework for API

TensorFlow / Keras → Deep learning model

scikit-learn → StandardScaler for preprocessing

joblib → Saving/loading scaler

numpy & pandas → Data processing

(See requirements.txt for full list)

🧑‍💻 Training the Model

Prepare dataset (df) where the target column is new_cases.

Split into features X and target y:

```python
X = df.drop(columns=["new_cases"])
y = df["new_cases"]
```

Scale features with StandardScaler, train/test split, and fit the model:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)


Save model & scaler for deployment:

```python
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
```

🌐 Running the Flask API

Start the Flask server:

```bash
python app.py
```

By default, it runs at:

http://127.0.0.1:5000

📡 API Endpoints
POST /predict

Make a prediction by sending a JSON array of 91 features (must match training).

Request Example
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features":[0.25,-0.13,0.67,1.42,-0.98,0.33,0.81,-0.57,1.19,-0.44,
                      0.73,-1.02,0.28,0.56,-0.39,1.07,-0.88,0.14,0.62,-0.21,
                      0.35,0.47,-0.65,0.92,-0.18,1.26,0.41,-0.53,0.87,-0.74,
                      0.52,0.11,-0.83,1.31,0.69,-0.36,0.58,-1.09,0.77,-0.25,
                      1.04,0.32,-0.41,0.63,0.19,-0.72,0.96,-0.11,0.84,-0.47,
                      0.23,0.55,-0.66,0.91,0.37,-0.29,1.12,-0.85,0.46,0.71,
                      -0.34,0.59,-1.21,0.28,0.93,-0.48,0.62,-0.19,0.51,0.12,
                      -0.77,1.08,0.45,-0.23,0.64,0.89,-0.52,0.17,0.73,-0.68,
                      0.39,1.15,-0.14,0.82,-0.91]}'
```

Response Example
```bash
{
  "prediction": 1234.56
}
```

🧪 Testing with Python

You can also test with a Python client:

```python
import requests
import numpy as np

url = "http://127.0.0.1:5000/predict"

# Generate a fake 91-feature vector
features = list(np.random.randn(91))

response = requests.post(url, json={"features": features})
print(response.json())
```

🚀 Deployment Options

Localhost (default for testing)

Gunicorn + Nginx (production-ready API)

Docker (containerized deployment)

Heroku / Render / Azure / AWS (cloud deployment)

📌 Notes

Make sure your request always contains exactly 91 features.

If feature mismatch occurs, check:
- Dataset preprocessing consistency
- Scaler.pkl alignment with model.h5
