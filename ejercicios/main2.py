# Importar las librerías necesarias
import pandas as pd
import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from kan import KAN
from kan.utils import ex_round

# Establecer la semilla para reproducibilidad
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configurar el dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Supongamos que tu dataframe se llama 'df' y tiene las columnas mencionadas
# Para este ejemplo, crearemos un dataframe de ejemplo
num_samples = 10000
time = np.arange(num_samples)
data = np.random.normal(
    0, 1, (num_samples, 12)
)  # 12 columnas para 4 sensores (X, Y, Z cada uno)
columns = [
    "ax1",
    "ay1",
    "az1",
    "ax2",
    "ay2",
    "az2",
    "ax3",
    "ay3",
    "az3",
    "ax4",
    "ay4",
    "az4",
]
df = pd.DataFrame(data, columns=columns)
df["time"] = time
df = df[["time"] + columns]

# Preprocesamiento
# Eliminamos la columna 'time' para el entrenamiento
features = df.drop("time", axis=1)

# Normalizamos las características
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Convertimos los datos a tensores de PyTorch
X = torch.tensor(features_scaled, dtype=torch.float32).to(device)

# Dividimos los datos en conjuntos de entrenamiento, validación y prueba
X_temp, X_test = train_test_split(X.cpu().numpy(), test_size=0.2, random_state=seed)
X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=seed)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

# Crear datasets y dataloaders para el autoencoder tradicional
batch_size = 64

train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
test_dataset = torch.utils.data.TensorDataset(X_test, X_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Definir el Autoencoder Tradicional
class TraditionalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TraditionalAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


input_dim = X_train.shape[1]

# Inicializar el modelo y moverlo al dispositivo
traditional_model = TraditionalAutoencoder(input_dim).to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(traditional_model.parameters(), lr=1e-3)

# Implementar Early Stopping
num_epochs = 100
patience = 10
best_val_loss = np.inf
epochs_no_improve = 0
best_model_wts = None

for epoch in range(num_epochs):
    traditional_model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = traditional_model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validación
    traditional_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = traditional_model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}"
    )

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_wts = traditional_model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping!")
            # Cargar los mejores pesos
            traditional_model.load_state_dict(best_model_wts)
            break

# -------------------------------
# Evaluación y Detección de Anomalías con el Autoencoder Tradicional
# -------------------------------
traditional_model.eval()
reconstruction_errors = []
with torch.no_grad():
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = traditional_model(inputs)
        loss = torch.mean((outputs - inputs) ** 2, dim=1)
        reconstruction_errors.extend(loss.cpu().numpy())

# Establecer un umbral para la detección de anomalías
threshold = np.percentile(reconstruction_errors, 95)  # Por ejemplo, percentil 95

# Detectar anomalías
anomalies = np.where(reconstruction_errors > threshold)[0]
print(
    f"Número de anomalías detectadas con el Autoencoder Tradicional: {len(anomalies)}"
)

# -------------------------------
# Predicción Conformal para el Autoencoder Tradicional
# -------------------------------
# Calculamos los errores de calibración en el conjunto de entrenamiento
calibration_errors = []
with torch.no_grad():
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = traditional_model(inputs)
        loss = torch.mean((outputs - inputs) ** 2, dim=1)
        calibration_errors.extend(loss.cpu().numpy())

# Establecer el nivel de significancia
alpha = 0.05  # Confianza del 95%

# Determinar el umbral de cuantiles a partir de los errores de calibración
quantile = np.quantile(calibration_errors, 1 - alpha)

# Aplicar el umbral a los datos de prueba
anomalies_conformal = np.where(reconstruction_errors > quantile)[0]
print(
    f"Número de anomalías detectadas con el Autoencoder Tradicional usando predicción conformal: {len(anomalies_conformal)}"
)

# -------------------------------
# Implementación del Autoencoder basado en KAN
# -------------------------------
# Crear dataset compatible con KAN
dataset = {
    "train_input": X_train,
    "train_label": X_train,  # En un autoencoder, la salida es igual a la entrada
    "test_input": X_test,
    "test_label": X_test,
}

# Crear el modelo KAN
# Ajusta 'width', 'grid', 'k' según tus necesidades
model = KAN(width=[input_dim, 16, input_dim], grid=5, k=3, seed=seed, device=device)

# Asegurarnos de que el modelo KAN utilice torch.float32
torch.set_default_dtype(torch.float32)

# Entrenar el modelo con Early Stopping
best_val_loss_kan = np.inf
epochs_no_improve_kan = 0
best_model_wts_kan = None

for epoch in range(num_epochs):
    model.fit(dataset, opt="LBFGS", steps=1, lamb=0.001)

    # Validación
    with torch.no_grad():
        outputs = model(dataset["test_input"])
        val_loss = torch.mean((outputs - dataset["test_label"]) ** 2).item()

    print(f"Epoch {epoch+1}/{num_epochs}, KAN Validation Loss: {val_loss:.6f}")

    # Early Stopping
    if val_loss < best_val_loss_kan:
        best_val_loss_kan = val_loss
        epochs_no_improve_kan = 0
        # Guardar los mejores pesos (parámetros del modelo)
        best_model_wts_kan = model.state_dict()
    else:
        epochs_no_improve_kan += 1
        if epochs_no_improve_kan >= patience:
            print("Early stopping para KAN!")
            # Cargar los mejores pesos
            model.load_state_dict(best_model_wts_kan)
            break

# -------------------------------
# Evaluación y Detección de Anomalías con KAN
# -------------------------------
with torch.no_grad():
    outputs = model(dataset["test_input"])
    reconstruction_errors_kan = (
        torch.mean((outputs - dataset["test_label"]) ** 2, dim=1).cpu().numpy()
    )

# Establecer un umbral para la detección de anomalías
threshold_kan = np.percentile(reconstruction_errors_kan, 95)

# Detectar anomalías
anomalies_kan = np.where(reconstruction_errors_kan > threshold_kan)[0]
print(f"Número de anomalías detectadas con KAN: {len(anomalies_kan)}")

# -------------------------------
# Predicción Conformal para KAN
# -------------------------------
with torch.no_grad():
    outputs_train = model(dataset["train_input"])
    calibration_errors_kan = (
        torch.mean((outputs_train - dataset["train_label"]) ** 2, dim=1).cpu().numpy()
    )

# Determinar el umbral de cuantiles a partir de los errores de calibración
quantile_kan = np.quantile(calibration_errors_kan, 1 - alpha)

# Aplicar el umbral a los datos de prueba
anomalies_conformal_kan = np.where(reconstruction_errors_kan > quantile_kan)[0]
print(
    f"Número de anomalías detectadas con KAN usando predicción conformal: {len(anomalies_conformal_kan)}"
)
