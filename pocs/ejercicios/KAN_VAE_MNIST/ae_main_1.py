# Importar módulos necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
# %matplotlib inline

# Establecer semilla para reproducibilidad
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Generar datos sintéticos para 3 sensores
def generate_sensor_data(sensor_id, num_samples=2000, anomaly=False):
    np.random.seed(seed + sensor_id)
    timestamps = np.arange(num_samples)
    # Datos normales: aceleraciones con ruido gaussiano
    accel_x = np.sin(0.02 * timestamps) + np.random.normal(0, 0.1, num_samples)
    accel_y = np.cos(0.02 * timestamps) + np.random.normal(0, 0.1, num_samples)
    accel_z = np.sin(0.02 * timestamps) * np.cos(0.02 * timestamps) + np.random.normal(
        0, 0.1, num_samples
    )
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
        }
    )
    if anomaly:
        # Introducir anomalías en una porción de los datos
        anomaly_indices = np.random.choice(
            num_samples, size=int(0.9 * num_samples), replace=False
        )
        data.loc[anomaly_indices, ["accel_x", "accel_y", "accel_z"]] += (
            np.random.normal(0, 3, (len(anomaly_indices), 3))
        )
    return data


# Generar datos para los 3 sensores
sensor_data = []
for i in range(3):
    data = generate_sensor_data(
        sensor_id=i, num_samples=2000, anomaly=(i == 2)
    )  # El sensor 3 tendrá anomalías
    data["sensor"] = f"sensor_{i+1}"
    sensor_data.append(data)

# Concatenar los datos de los 3 sensores
combined_data = pd.concat(sensor_data, ignore_index=True)

# Normalizar los datos de aceleración
scaler = MinMaxScaler()
combined_data[["accel_x", "accel_y", "accel_z"]] = scaler.fit_transform(
    combined_data[["accel_x", "accel_y", "accel_z"]]
)


# Convertir los datos en tensores y crear un dataset personalizado
class SensorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.X = torch.tensor(
            data[["accel_x", "accel_y", "accel_z"]].values, dtype=torch.float32
        )
        self.labels = (
            data["sensor"].apply(lambda x: 1 if x == "sensor_3" else 0).values
        )  # Anomalías en sensor_3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


dataset = SensorDataset(combined_data)

# Porcentaje de división
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Dividir el dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed),
)

# Crear DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Definir el autoencoder basado en LSTM
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((x.shape[0], self.seq_len, self.n_features))
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        output, (hidden, cell) = self.decoder(hidden)
        return output


# Parámetros del modelo
seq_len = 1  # Longitud de la secuencia
n_features = 3  # Número de características (accel_x, accel_y, accel_z)
embedding_dim = 64  # Dimensión del espacio latente

model = LSTMAutoencoder(seq_len, n_features, embedding_dim)

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Implementación de early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.min_delta = min_delta
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Resetear contador si hay mejora
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Función de entrenamiento
def train_model(model, train_loader, val_loader, num_epochs=100):
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            X_pred = model(X_batch)
            loss = criterion(X_pred, X_batch.reshape(X_pred.shape))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        history["train_loss"].append(train_loss)

        # Validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_pred = model(X_batch)
                loss = criterion(X_pred, X_batch.reshape(X_pred.shape))
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Pérdida de Entrenamiento: {train_loss:.6f}, Pérdida de Validación: {val_loss:.6f}"
        )

        # Comprobar early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Detención temprana en la época {epoch+1}")
            break
    return history


# Entrenar el modelo
start_time = time.time()
history = train_model(model, train_loader, val_loader, num_epochs=100)
end_time = time.time()
print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")


# Obtener los errores de reconstrucción en el conjunto de validación
def get_reconstruction_errors(model, loader):
    errors = []
    targets = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_pred = model(X_batch)
            loss = nn.functional.mse_loss(
                X_pred, X_batch.reshape(X_pred.shape), reduction="none"
            )
            loss = loss.view(loss.size(0), -1).mean(
                dim=1
            )  # Asegurar que loss es de forma (batch_size,)
            errors.extend(loss.cpu().numpy())
            targets.extend(y_batch.numpy())
    return np.array(errors), np.array(targets)


# Calcular errores en el conjunto de validación
val_errors, val_targets = get_reconstruction_errors(model, val_loader)

# Establecer nivel de significancia alpha
alpha = 0.05

# Calcular umbral de conformidad
threshold = np.quantile(val_errors, 1 - alpha)
print(f"Umbral de detección (Conformal Prediction) con alpha={alpha}: {threshold}")

# Detectar anomalías en el conjunto de prueba
test_errors, test_targets = get_reconstruction_errors(model, test_loader)
test_predictions = (test_errors > threshold).astype(
    int
)  # 1 si es anomalía, 0 si es normal

# Analizar los resultados
results_df = pd.DataFrame(
    {
        "Error_de_reconstrucción": test_errors,
        "Anomalía_predicha": test_predictions,
        "Anomalía_real": test_targets,
    }
)

# Evaluación del modelo
print("Reporte de clasificación:")
print(
    classification_report(results_df["Anomalía_real"], results_df["Anomalía_predicha"])
)

print("Matriz de confusión:")
print(confusion_matrix(results_df["Anomalía_real"], results_df["Anomalía_predicha"]))

# Histograma de errores de reconstrucción
plt.figure(figsize=(10, 6))
normal_errors = results_df[results_df["Anomalía_real"] == 0]["Error_de_reconstrucción"]
anomaly_errors = results_df[results_df["Anomalía_real"] == 1]["Error_de_reconstrucción"]
plt.hist(normal_errors, bins=50, alpha=0.5, label="Datos Normales")
plt.hist(anomaly_errors, bins=50, alpha=0.5, label="Anomalías")
plt.axvline(
    x=threshold, color="r", linestyle="--", label="Umbral de Conformal Prediction"
)
plt.legend()
plt.xlabel("Error de Reconstrucción (MSE)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores de Reconstrucción")
plt.show()

# Extraer embeddings del conjunto de prueba
embeddings = []
targets = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.reshape((X_batch.shape[0], seq_len, n_features))
        _, (hidden, _) = model.encoder(X_batch)
        embeddings.extend(hidden.squeeze(0).cpu().numpy())
        targets.extend(y_batch.numpy())

embeddings = np.array(embeddings)
targets = np.array(targets)

# Visualización en 2D usando PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(
    embeddings_2d[targets == 0, 0],
    embeddings_2d[targets == 0, 1],
    label="Datos Normales",
    alpha=0.5,
)
plt.scatter(
    embeddings_2d[targets == 1, 0],
    embeddings_2d[targets == 1, 1],
    label="Anomalías",
    alpha=0.5,
)
plt.legend()
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Visualización de Embeddings")
plt.show()
