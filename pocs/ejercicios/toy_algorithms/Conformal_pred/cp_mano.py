import os
import numpy as np
import pickle
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

n_samples = 5000
n_new = 350

# Generamos el dataset de dos lunas
dataset = 4 * make_moons(n_samples=n_samples, noise=0.05, random_state=42)[0] - np.array([0.5, 0.25])

# Generamos nuevos puntos de datos (prueba) de forma uniforme
rng = np.random.RandomState(42)
z_test = rng.uniform(low=-6, high=10, size=(n_new, 2))

class ADPredictor:
    def __init__(self, model):
        self.model = model

    def predict_scores(self, X):
        return -self.model.decision_function(X)

# Implementación del Conformal Anomaly Detector sin usar puncc
class ConformalAnomalyDetector:
    def __init__(self, predictor, fit_ratio=0.7, random_state=None):
        self.predictor = predictor
        self.fit_ratio = fit_ratio
        self.random_state = random_state
        self.threshold = None

    def fit(self, z, alpha=0.05):
        # Dividir los datos en entrenamiento y calibración
        self.train_data, self.calibration_data = train_test_split(
            z, train_size=self.fit_ratio, random_state=self.random_state
        )
        # Ajustar el modelo en los datos de entrenamiento
        self.predictor.model.fit(self.train_data)
        # Obtener las puntuaciones de los datos de calibración
        calibration_scores = self.predictor.predict_scores(self.calibration_data)
        # Determinar el umbral basado en el percentil (1 - alpha)
        self.threshold = np.quantile(calibration_scores, 1 - alpha)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("El modelo no ha sido ajustado. Llame a 'fit' primero.")
        scores = self.predictor.predict_scores(X)
        anomalies = scores > self.threshold
        return anomalies

def main():
    # Inicializar el modelo de detección de anomalías
    ad_model = IsolationForest(random_state=42)

    # Envolver el modelo de detección de anomalías en el predictor
    if_predictor = ADPredictor(ad_model)

    # Inicializar el Conformal Anomaly Detector
    cad = ConformalAnomalyDetector(if_predictor, fit_ratio=0.7, random_state=0)

    # Ajustar el CAD en el dataset
    alpha = 0.05
    cad.fit(dataset, alpha=alpha)

    # Predecir anomalías en los datos de prueba
    cad_results = cad.predict(z_test)
    cad_anomalies = z_test[cad_results]
    cad_not_anomalies = z_test[~cad_results]

    # Detectar anomalías con el modelo IF subyacente (sin conformal)
    if_predictor.model.fit(dataset)
    if_results = if_predictor.model.predict(z_test) == -1  # En IsolationForest, -1 indica anomalía
    if_anomalies = z_test[if_results]
    if_not_anomalies = z_test[~if_results]

    # Ruta para guardar los resultados
    results_path = "cp_mano_result.pkl"

    # Asegurarse de que el directorio existe
    #os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Guardar los resultados utilizando pickle
    results = {
        "cad_results": cad_results,
        "cad_anomalies": cad_anomalies,
        "cad_not_anomalies": cad_not_anomalies,
        "if_results": if_results,
        "if_anomalies": if_anomalies,
        "if_not_anomalies": if_not_anomalies
    }

    with open(results_path, "wb") as file:
        pickle.dump(results, file)

    print(f"Resultados guardados en {results_path}")

    # Opcional: Guardar las visualizaciones como imágenes
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

    # Plot de los resultados de Isolation Forest
    ax[0].scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers")
    ax[0].scatter(
        if_not_anomalies[:, 0],
        if_not_anomalies[:, 1],
        s=40,
        marker="x",
        color="blue",
        label="Normal",
    )
    ax[0].scatter(
        if_anomalies[:, 0],
        if_anomalies[:, 1],
        s=40,
        marker="x",
        color="red",
        label="Anomalía",
    )
    ax[0].set_xticks(())
    ax[0].set_yticks(())
    ax[0].set_title("Isolation Forest")
    ax[0].legend(loc="lower left")

    # Plot de los resultados de Conformalized Isolation Forest
    ax[1].scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers")
    ax[1].scatter(
        cad_not_anomalies[:, 0],
        cad_not_anomalies[:, 1],
        marker="x",
        color="blue",
        s=40,
        label="Normal",
    )
    ax[1].scatter(
        cad_anomalies[:, 0],
        cad_anomalies[:, 1],
        marker="x",
        color="red",
        s=40,
        label="Anomalía",
    )
    ax[1].set_xticks(())
    ax[1].set_yticks(())
    ax[1].set_title("Conformalized Isolation Forest")
    ax[1].legend(loc="lower left")

    # Guardar las visualizaciones
    plt.savefig("anomaly_results.png")
    plt.close()
    print("Visualizaciones guardadas en 'anomaly_results.png'")

if __name__ == "__main__":
    main()