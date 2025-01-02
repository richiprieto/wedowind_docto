import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Tuple, List

def generate_time_series(n_samples: int = 1000, contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una serie temporal simulada con anomalías.
    
    :param n_samples: Número total de muestras
    :param contamination: Fracción de anomalías
    :return: X (serie temporal), y (etiquetas: 0 normal, 1 anomalía)
    """
    t = np.linspace(0, 10, n_samples)
    X = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    
    # Agregar anomalías
    n_anomalies = int(contamination * n_samples)
    anomaly_idx = np.random.choice(n_samples, n_anomalies, replace=False)
    X[anomaly_idx] += np.random.normal(0, 0.5, n_anomalies)
    
    y = np.zeros(n_samples)
    y[anomaly_idx] = 1
    
    return X.reshape(-1, 1), y

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias de una serie temporal para el entrenamiento.
    
    :param X: Serie temporal
    :param y: Etiquetas
    :param seq_length: Longitud de la secuencia
    :return: Secuencias X, etiquetas y
    """
    sequences = []
    labels = []
    
    for i in range(len(X) - seq_length + 1):
        seq = X[i:i+seq_length]
        label = y[i+seq_length-1]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def conformal_anomaly_detection(X_train: np.ndarray, X_cal: np.ndarray, X_test: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza detección de anomalías usando Conformal Prediction.
    
    :param X_train: Datos de entrenamiento
    :param X_cal: Datos de calibración 
    :param X_test: Datos de prueba
    :param alpha: Nivel de significancia (1 - confianza)
    :return: predicciones, puntuaciones_conformidad
    """
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_train.reshape(X_train.shape[0], -1))
    
    cal_scores = -clf.score_samples(X_cal.reshape(X_cal.shape[0], -1))
    test_scores = -clf.score_samples(X_test.reshape(X_test.shape[0], -1))
    
    threshold = np.percentile(cal_scores, (1 - alpha) * 100)
    predictions = test_scores > threshold
    
    return predictions, test_scores

def plot_results(X: np.ndarray, y: np.ndarray, predictions: np.ndarray, scores: np.ndarray):
    """
    Visualiza los resultados de la detección de anomalías en la serie temporal.
    
    :param X: Serie temporal
    :param y: Etiquetas verdaderas
    :param predictions: Predicciones del modelo
    :param scores: Puntuaciones de conformidad
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(211)
    plt.plot(X, label='Serie Temporal')
    anomalies = X[y == 1]
    plt.scatter(np.where(y == 1)[0], anomalies, color='red', label='Anomalías Reales')
    plt.title('Serie Temporal con Anomalías Reales')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(X, label='Serie Temporal')
    detected_anomalies = X[predictions == 1]
    plt.scatter(np.where(predictions == 1)[0], detected_anomalies, color='green', label='Anomalías Detectadas')
    plt.title('Serie Temporal con Anomalías Detectadas')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de rendimiento para la detección de anomalías.
    
    :param y_true: Etiquetas verdaderas
    :param y_pred: Predicciones
    :return: Diccionario con métricas de rendimiento
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }

def main():
    # Generar serie temporal
    X, y = generate_time_series(n_samples=1000, contamination=0.1)
    
    # Crear secuencias
    seq_length = 1
    X_seq, y_seq = create_sequences(X, y, seq_length)
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    train_size = int(0.6 * len(X_seq))
    val_size = int(0.2 * len(X_seq))
    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Aplicar Conformal Anomaly Detection
    predictions, scores = conformal_anomaly_detection(X_train_scaled, X_val_scaled, X_test_scaled, alpha=0.2)
    
    # Evaluar rendimiento
    performance = evaluate_performance(y_test, predictions)
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualizar resultados
    plot_results(X[train_size+val_size+seq_length-1:], y[train_size+val_size+seq_length-1:], predictions, scores)

if __name__ == "__main__":
    main()