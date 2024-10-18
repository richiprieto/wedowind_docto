import os
import numpy as np
import pickle
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.anomaly_detection import SplitCAD

# ----------------------------
# Common Data Generation
# ----------------------------
n_samples = 5000
n_new = 350

# Generate the two moons dataset
dataset = 4 * make_moons(n_samples=n_samples, noise=0.05, random_state=42)[0] - np.array([0.5, 0.25])

# Generate uniformly new (test) data points
rng = np.random.RandomState(42)
z_test = rng.uniform(low=-6, high=10, size=(n_new, 2))

# ----------------------------
# Assign True Labels for Evaluation
# ----------------------------
# Define a simple rule to assign true labels
# Points within the range of the training data are considered normal (0), others anomalies (1)
x_min, y_min = dataset.min(axis=0) - 1
x_max, y_max = dataset.max(axis=0) + 1

def assign_true_labels(z):
    return np.where(
        (z[:, 0] >= x_min) & (z[:, 0] <= x_max) &
        (z[:, 1] >= y_min) & (z[:, 1] <= y_max),
        0, 1
    )

true_labels = assign_true_labels(z_test)

# ----------------------------
# Isolation Forest Model
# ----------------------------
ad_model = IsolationForest(random_state=42)

# ----------------------------
# Predictor for deel.puncc (SplitCAD)
# ----------------------------
class ADPredictor(BasePredictor):
    def predict(self, X):
        return -self.model.score_samples(X)

# Wrap the Isolation Forest model in the predictor
if_predictor = ADPredictor(ad_model)

# Instantiate SplitCAD on top of the predictor
split_cad = SplitCAD(if_predictor, train=True, random_state=0)

# Fit SplitCAD on the dataset with a fit ratio of 0.7
split_cad.fit(z=dataset, fit_ratio=0.7)

# Predict anomalies using SplitCAD with alpha = 0.05
alpha = 0.05
split_cad_results = split_cad.predict(z_test, alpha=alpha)
split_cad_anomalies = z_test[split_cad_results]
split_cad_not_anomalies = z_test[np.invert(split_cad_results)]

# ----------------------------
# Manual Conformal Anomaly Detector
# ----------------------------
class ManualADPredictor:
    def __init__(self, model):
        self.model = model

    def predict_scores(self, X):
        return -self.model.decision_function(X)

class ConformalAnomalyDetector:
    def __init__(self, predictor, fit_ratio=0.7, random_state=None):
        self.predictor = predictor
        self.fit_ratio = fit_ratio
        self.random_state = random_state
        self.threshold = None

    def fit(self, z, alpha=0.05):
        # Split the data into training and calibration sets
        self.train_data, self.calibration_data = train_test_split(
            z, train_size=self.fit_ratio, random_state=self.random_state
        )
        # Fit the model on the training data
        self.predictor.model.fit(self.train_data)
        # Get the scores for calibration data
        calibration_scores = self.predictor.predict_scores(self.calibration_data)
        # Determine the threshold based on the (1 - alpha) quantile
        self.threshold = np.quantile(calibration_scores, 1 - alpha)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("The model has not been fitted. Call 'fit' first.")
        scores = self.predictor.predict_scores(X)
        anomalies = scores > self.threshold
        return anomalies

# Initialize the manual predictor and Conformal Anomaly Detector
manual_predictor = ManualADPredictor(ad_model)
manual_cad = ConformalAnomalyDetector(manual_predictor, fit_ratio=0.7, random_state=0)

# Fit the manual CAD on the dataset
manual_cad.fit(z=dataset, alpha=alpha)

# Predict anomalies using the manual CAD
manual_cad_results = manual_cad.predict(z_test)
manual_cad_anomalies = z_test[manual_cad_results]
manual_cad_not_anomalies = z_test[~manual_cad_results]

# ----------------------------
# Standard Isolation Forest Predictions
# ----------------------------
# Fit the Isolation Forest on the entire dataset
ad_model.fit(dataset)

# Predict anomalies using Isolation Forest (without conformal)
if_results = ad_model.predict(z_test) == -1  # -1 indicates anomaly
if_anomalies = z_test[if_results]
if_not_anomalies = z_test[~if_results]

# ----------------------------
# Evaluation Metrics
# ----------------------------
from sklearn.metrics import classification_report

# Predictions for each method
pred_if = if_results.astype(int)
pred_split_cad = split_cad_results.astype(int)
pred_manual_cad = manual_cad_results.astype(int)

# Generate classification reports
report_if = classification_report(true_labels, pred_if, output_dict=True)
report_split_cad = classification_report(true_labels, pred_split_cad, output_dict=True)
report_manual_cad = classification_report(true_labels, pred_manual_cad, output_dict=True)

print(report_if)
print(report_split_cad)
print(report_manual_cad)

# ----------------------------
# Comparison Table
# ----------------------------
comparison_data = {
    "Method": [
        "Isolation Forest",
        "SplitCAD (deel.puncc)",
        "Manual Conformal CAD"
    ],
    "Total Anomalies Detected": [
        if_anomalies.shape[0],
        split_cad_anomalies.shape[0],
        manual_cad_anomalies.shape[0]
    ],
    "Normal Points Detected": [
        if_not_anomalies.shape[0],
        split_cad_not_anomalies.shape[0],
        manual_cad_not_anomalies.shape[0]
    ],
    "Precision": [
        f"{report_if['1']['precision'] * 100:.2f}",
        f"{report_split_cad['1']['precision'] * 100:.2f}",
        f"{report_manual_cad['1']['precision'] * 100:.2f}"
    ],
    "Recall": [
        f"{report_if['1']['recall'] * 100:.2f}",
        f"{report_split_cad['1']['recall'] * 100:.2f}",
        f"{report_manual_cad['1']['recall'] * 100:.2f}"
    ],
    "F1-Score": [
        f"{report_if['1']['f1-score'] * 100:.2f}",
        f"{report_split_cad['1']['f1-score'] * 100:.2f}",
        f"{report_manual_cad['1']['f1-score'] * 100:.2f}"
    ],
    "Support": [
        int(report_if['1']['support']),
        int(report_split_cad['1']['support']),
        int(report_manual_cad['1']['support'])
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("### Comparison of Anomaly Detection Methods\n")
print(comparison_df.to_markdown(index=False))

# ----------------------------
# Visualization
# ----------------------------
fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharex=True, sharey=True)

# Plot Isolation Forest results
ax[0].scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers", alpha=0.5)
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
    label="Anomaly",
)
ax[0].set_title("Isolation Forest")
ax[0].legend(loc="lower left")
ax[0].set_xticks([])
ax[0].set_yticks([])

# Plot SplitCAD results
ax[1].scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers", alpha=0.5)
ax[1].scatter(
    split_cad_not_anomalies[:, 0],
    split_cad_not_anomalies[:, 1],
    marker="x",
    color="blue",
    s=40,
    label="Normal",
)
ax[1].scatter(
    split_cad_anomalies[:, 0],
    split_cad_anomalies[:, 1],
    marker="x",
    color="red",
    s=40,
    label="Anomaly",
)
ax[1].set_title("SplitCAD (deel.puncc)")
ax[1].legend(loc="lower left")
ax[1].set_xticks([])
ax[1].set_yticks([])

# Plot Manual Conformal CAD results
ax[2].scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers", alpha=0.5)
ax[2].scatter(
    manual_cad_not_anomalies[:, 0],
    manual_cad_not_anomalies[:, 1],
    marker="x",
    color="blue",
    s=40,
    label="Normal",
)
ax[2].scatter(
    manual_cad_anomalies[:, 0],
    manual_cad_anomalies[:, 1],
    marker="x",
    color="red",
    s=40,
    label="Anomaly",
)
ax[2].set_title("Manual Conformal CAD")
ax[2].legend(loc="lower left")
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.tight_layout()
plt.show()

# ----------------------------
# Saving Results
# ----------------------------
results_path = "compare_conformal_results.pkl"

# Prepare results dictionary
results = {
    "if_results": if_results,
    "if_anomalies": if_anomalies,
    "if_not_anomalies": if_not_anomalies,
    "split_cad_results": split_cad_results,
    "split_cad_anomalies": split_cad_anomalies,
    "split_cad_not_anomalies": split_cad_not_anomalies,
    "manual_cad_results": manual_cad_results,
    "manual_cad_anomalies": manual_cad_anomalies,
    "manual_cad_not_anomalies": manual_cad_not_anomalies,
    "true_labels": true_labels
}

# Save the results using pickle
with open(results_path, "wb") as file:
    pickle.dump(results, file)

print(f"Resultados guardados en {results_path}")

# Save the visualization as an image
plt.savefig("anomaly_comparison_results.png")
plt.close()
print("Visualizaciones guardadas en 'anomaly_comparison_results.png'")
