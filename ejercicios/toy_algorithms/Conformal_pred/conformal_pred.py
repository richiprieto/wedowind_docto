import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

n_samples = 5000
n_new = 350

# We generate the two moons dataset
dataset = 4 * make_moons(n_samples=n_samples, noise=0.05, random_state=42)[
    0
] - np.array([0.5, 0.25])

# We generate uniformly new (test) data points
rng = np.random.RandomState(42)
z_test = rng.uniform(low=-6, high=10, size=(n_new, 2))

from sklearn.ensemble import IsolationForest

ad_model = IsolationForest(random_state=42) 

# Note that any other model could have been used, such as LOF:
# from sklearn.neighbors import LocalOutlierFactor
# ad_model = LocalOutlierFactor(n_neighbors=35, novelty=True)

from deel.puncc.api.prediction import BasePredictor

# We redefine the predict method to return the opposite of IF scores
class ADPredictor(BasePredictor):
    def predict(self, X):
        return -self.model.score_samples(X)

# wrap the (IF) anomaly detection model in a predictor
if_predictor = ADPredictor(ad_model)

from deel.puncc.anomaly_detection import SplitCAD

# Instantiate CAD on top of IF predictor
if_cad = SplitCAD(if_predictor, train=True, random_state=0)

# Fit the IF on the proper fitting dataset and
# calibrate it using calibration dataset.
# The two datasets are sampled randomly with a ration of 7:3,
# respectively.
if_cad.fit(z=dataset, fit_ratio=0.7)

# We set the maximum false detection rate to 5%
alpha = 0.05
# The method `predict` is called on the new data points
# to test which are anomalous and which are not
cad_results = if_cad.predict(z_test, alpha=alpha)
cad_anomalies = z_test[cad_results]
cad_not_anomalies = z_test[np.invert(cad_results)]

# Detect anomalies with underlying IF model (no conformal)
if_results = if_cad.predictor.model.predict(z_test) == 1
if_not_anomalies = z_test[if_results]
if_anomalies = z_test[np.invert(if_results)]

fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

# Plot if results

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
    label="Anomaly",
)
ax[0].set_xticks(())
ax[0].set_yticks(())
ax[0].set_title("Isolation Forest")
ax[0].legend(loc="lower left")

# Plot cad results
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
    label="Anomaly",
)
ax[1].set_xticks(())
ax[1].set_yticks(())
ax[1].set_title("Conformalized Isolation Forest")
ax[1].legend(loc="lower left")

plt.show()