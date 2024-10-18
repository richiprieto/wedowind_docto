import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

def generate_data(n_samples=1000, contamination=0.1):
    """
    Genera datos simulados con anomalías.
    
    :param n_samples: Número total de muestras
    :param contamination: Fracción de anomalías
    :return: X (datos), y (etiquetas: 0 normal, 1 anomalía)
    """
    n_inliers = int((1 - contamination) * n_samples)
    n_outliers = n_samples - n_inliers
    
    # Generar datos normales
    X_inliers = np.random.randn(n_inliers, 2)
    
    # Generar anomalías
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
    
    # Combinar datos normales y anomalías
    X = np.vstack([X_inliers, X_outliers])
    y = np.zeros(n_samples)
    y[n_inliers:] = 1  # Marcar anomalías
    
    return X, y

def conformal_anomaly_detection(X_train, X_test, alpha=0.1):
    """
    Realiza detección de anomalías usando Conformal Prediction.
    
    :param X_train: Datos de entrenamiento
    :param X_test: Datos de prueba
    :param alpha: Nivel de significancia (1 - confianza)
    :return: predicciones, puntuaciones_conformidad
    """
    # Usar Isolation Forest como detector de anomalías base
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_train)
    
    # Calcular puntuaciones de conformidad para el conjunto de calibración (entrenamiento)
    cal_scores = -clf.score_samples(X_train)
    
    # Calcular puntuaciones de conformidad para el conjunto de prueba
    test_scores = -clf.score_samples(X_test)
    
    # Calcular el umbral de conformidad
    threshold = np.percentile(cal_scores, (1 - alpha) * 100)
    
    # Realizar predicciones
    predictions = test_scores > threshold
    
    return predictions, test_scores

def plot_results(X, y, predictions, scores):
    """
    Visualiza los resultados de la detección de anomalías.
    
    :param X: Datos
    :param y: Etiquetas verdaderas
    :param predictions: Predicciones del modelo
    :param scores: Puntuaciones de conformidad
    """
    plt.figure(figsize=(12, 5))
    
    # Graficar datos originales
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title("Datos Originales")
    plt.colorbar(label='Etiqueta (0: Normal, 1: Anomalía)')
    
    # Graficar predicciones
    plt.subplot(122)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Puntuación de Conformidad')
    plt.title("Predicciones de Anomalías")
    
    # Resaltar predicciones de anomalías
    anomalies = X[predictions == 1]
    plt.scatter(anomalies[:, 0], anomalies[:, 1], facecolors='none', edgecolors='r', s=100, label='Anomalías Detectadas')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Generar datos
    X, y = generate_data(n_samples=1000, contamination=0.1)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Aplicar Conformal Anomaly Detection
    predictions, scores = conformal_anomaly_detection(X_train, X_test, alpha=0.1)
    
    # Calcular métricas de rendimiento
    true_anomalies = y_test == 1
    detected_anomalies = predictions == 1
    
    accuracy = np.mean(true_anomalies == detected_anomalies)
    precision = np.sum(true_anomalies & detected_anomalies) / np.sum(detected_anomalies)
    recall = np.sum(true_anomalies & detected_anomalies) / np.sum(true_anomalies)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Visualizar resultados
    plot_results(X_test, y_test, predictions, scores)

if __name__ == "__main__":
    main()
