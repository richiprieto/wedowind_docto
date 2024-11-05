# ----------------------------
# Manual Conformal Anomaly Detector
# ----------------------------
import numpy as np
import torch

class ManualADPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(torch.tensor(data, dtype=torch.float32))
            loss = torch.mean((reconstructed - torch.tensor(data, dtype=torch.float32))**2, dim=1)
        return loss.numpy()

class ConformalAnomalyDetector:
    def __init__(self, predictor):
        self.predictor = predictor
        self.threshold = None

    def fit(self, train_data, calibration_data, alpha=0.05):
        # Generar scores de entrenamiento
        train_scores = self.predictor.predict(train_data)
        
        # Generar scores de calibración
        calibration_scores = self.predictor.predict(calibration_data)
        
        # Ajustar el umbral basado en los scores de calibración
        self.threshold = np.quantile(calibration_scores, 1 - alpha)
        return self.threshold

    def predict(self, data):
        scores = self.predictor.predict(data)
        anomalies = scores > self.threshold
        return anomalies
