from fastapi import FastAPI
import pandas as pd
import joblib
from scipy.stats import boxcox

app = FastAPI(title="Heart Disease Prediction API")

# ===============================
# LOAD ARTIFACTS
# ===============================
model = joblib.load("artifacts/svm_model.pkl")
lambdas = joblib.load("artifacts/boxcox_lambdas.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health():
    return {"status": "API running successfully"}

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict_heart_disease(data: dict):

    df = pd.DataFrame([data])

    # One-hot encoding
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_columns]

    # Box-Cox transformation
    for col, lam in lambdas.items():
        df[col] = boxcox(df[col], lmbda=lam)

    # Prediction
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
