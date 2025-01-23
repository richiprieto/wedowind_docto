# (Mantener igual que la versión anterior)
import numpy as np
import torch
import logging

class ManualADPredictor:
    def __init__(self, model):
        self.model = model
        logging.info("Predictor manual inicializado")

    def predict(self, data):
        try:
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(torch.tensor(data, dtype=torch.float32))
                loss = torch.mean((reconstructed - torch.tensor(data, dtype=torch.float32))**2, dim=1)
            return loss.numpy()
        except Exception as e:
            logging.error(f"Error en predicción: {str(e)}")
            raise

class ConformalAnomalyDetector:
    def __init__(self, predictor):
        self.predictor = predictor
        self.threshold = None
        logging.info("Conformal anomaly detector inicializado")

    def fit(self, train_data, calibration_data, alpha=0.05):
        try:
            calibration_scores = self.predictor.predict(calibration_data)
            self.threshold = np.quantile(calibration_scores, 1 - alpha)
            logging.info(f"Conformal threshold calculado: {self.threshold:.4f} (alpha={alpha})")
            return self.threshold
        except Exception as e:
            logging.error(f"Error in conformal fitting: {str(e)}")
            raise

    def predict(self, data):
        try:
            scores = self.predictor.predict(data)
            return scores > self.threshold
        except Exception as e:
            logging.error(f"Error in conformal detection: {str(e)}")
            raise
