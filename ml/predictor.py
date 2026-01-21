# backend/ml/predictor.py

import torch
import joblib
import numpy as np


autoencoder = torch.load("backend/ml/model.pt", map_location="cpu")
autoencoder.eval()

scaler = joblib.load("backend/ml/scaler.pkl")

def predict_risk(features: list) -> dict:
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        x = torch.tensor(X_scaled, dtype=torch.float32)
        reconstructed = autoencoder(x)
        error = torch.mean((x - reconstructed) ** 2).item()


    if error > 0.02:
        label = "HIGH RISK"
    else:
        label = "LOW RISK"

    return {
        "risk_score": round(error, 4),
        "risk_level": label
    }
