import pandas as pd
import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Establecer la semilla para reproducibilidad
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Asumimos que el dataframe ya está cargado como 'df'
# El dataframe tiene columnas: 'time', 'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'ax3', 'ay3', 'az3', 'ax4', 'ay4', 'az4'

# Para demostración, creamos un dataframe de ejemplo
# Reemplaza esto con tu código de carga de datos real
num_samples = 10000
time = np.arange(num_samples)
data = np.random.normal(0, 1, (num_samples, 12))
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
# Eliminamos la columna 'time' para entrenamiento
features = df.drop("time", axis=1)

# Normalizamos las características
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Dividimos los datos en conjuntos de entrenamiento, validación y prueba
X_temp, X_test = train_test_split(features_scaled, test_size=0.2, random_state=seed)
X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=seed)

# Convertimos los datos a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Creamos Datasets y DataLoaders
batch_size = 64

train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
test_dataset = torch.utils.data.TensorDataset(X_test, X_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Definimos el Autoencoder Tradicional
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


# Definimos el Autoencoder basado en KAN
class KANAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(KANAutoencoder, self).__init__()
        # Encoder basado en el teorema de Kolmogorov–Arnold
        # Aplicamos funciones univariadas a cada entrada
        self.psi_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(input_dim)
            ]
        )

        # Combinamos las salidas y aplicamos la función Φ
        self.phi_layer = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        # Aplicamos las funciones ψ_p a cada variable de entrada
        psi_outputs = []
        for i in range(x.size(1)):
            xi = x[:, i].unsqueeze(1)  # Forma: (batch_size, 1)
            psi_output = self.psi_layers[i](xi)  # Forma: (batch_size, hidden_dim)
            psi_outputs.append(psi_output)

        # Concatenamos las salidas
        psi_outputs = torch.cat(
            psi_outputs, dim=1
        )  # Forma: (batch_size, input_dim * hidden_dim)

        # Aplicamos la función Φ
        encoded = self.phi_layer(psi_outputs)  # Forma: (batch_size, hidden_dim)

        # Decodificamos
        decoded = self.decoder(encoded)  # Forma: (batch_size, input_dim)

        return decoded


input_dim = X_train.shape[1]

# Inicializamos ambos modelos
traditional_model = TraditionalAutoencoder(input_dim)
kan_model = KANAutoencoder(input_dim)

# Definimos funciones de pérdida y optimizadores para ambos modelos
criterion = nn.MSELoss()

traditional_optimizer = torch.optim.Adam(traditional_model.parameters(), lr=1e-3)
kan_optimizer = torch.optim.Adam(kan_model.parameters(), lr=1e-3)

# -------------------------------
# Entrenamiento del Autoencoder Tradicional con Early Stopping
# -------------------------------
num_epochs = 100
patience = 10  # Número de épocas para esperar mejora antes de detener
best_val_loss = np.inf
epochs_no_improve = 0
best_model_wts = None  # Para almacenar los pesos del mejor modelo

for epoch in range(num_epochs):
    traditional_model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        traditional_optimizer.zero_grad()
        outputs = traditional_model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        traditional_optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validación
    traditional_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
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
            # Cargamos los mejores pesos del modelo
            traditional_model.load_state_dict(best_model_wts)
            break

# -------------------------------
# Entrenamiento del Autoencoder KAN con Early Stopping
# -------------------------------
best_val_loss_kan = np.inf
epochs_no_improve_kan = 0
best_model_wts_kan = None  # Para almacenar los pesos del mejor modelo

for epoch in range(num_epochs):
    kan_model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        kan_optimizer.zero_grad()
        outputs = kan_model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        kan_optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validación
    kan_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            outputs = kan_model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, KAN Training Loss: {epoch_loss:.6f}, KAN Validation Loss: {val_loss:.6f}"
    )

    # Early Stopping
    if val_loss < best_val_loss_kan:
        best_val_loss_kan = val_loss
        epochs_no_improve_kan = 0
        best_model_wts_kan = kan_model.state_dict()
    else:
        epochs_no_improve_kan += 1
        if epochs_no_improve_kan >= patience:
            print("KAN Early stopping!")
            # Cargamos los mejores pesos del modelo
            kan_model.load_state_dict(best_model_wts_kan)
            break

# -------------------------------
# Evaluación y Detección de Anomalías
# -------------------------------


# Función para calcular errores de reconstrucción y detección de anomalías
def detect_anomalies(model, loader, alpha=0.05):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data in loader:
            inputs, _ = data
            outputs = model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.extend(loss.numpy())

    # Establecer un umbral para la detección de anomalías
    threshold = np.percentile(reconstruction_errors, 100 * (1 - alpha))

    # Detectar anomalías
    anomalies = np.where(reconstruction_errors > threshold)[0]
    num_anomalies = len(anomalies)
    return reconstruction_errors, anomalies, num_anomalies, threshold


# Detección de anomalías con el Autoencoder Tradicional
trad_recon_errors, trad_anomalies, trad_num_anomalies, trad_threshold = (
    detect_anomalies(traditional_model, test_loader)
)
print(
    f"Número de anomalías detectadas con el Autoencoder Tradicional: {trad_num_anomalies}"
)

# Detección de anomalías con el Autoencoder KAN
kan_recon_errors, kan_anomalies, kan_num_anomalies, kan_threshold = detect_anomalies(
    kan_model, test_loader
)
print(f"Número de anomalías detectadas con el Autoencoder KAN: {kan_num_anomalies}")

# -------------------------------
# Predicción Conformal para Detección de Anomalías
# -------------------------------


# Función para aplicar predicción conformal
def conformal_prediction(model, train_loader, test_recon_errors, alpha=0.05):
    model.eval()
    calibration_errors = []
    with torch.no_grad():
        for data in train_loader:
            inputs, _ = data
            outputs = model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=1)
            calibration_errors.extend(loss.numpy())

    # Determinar el umbral de cuantiles a partir de los errores de calibración
    calibration_errors = np.array(calibration_errors)
    quantile = np.quantile(calibration_errors, 1 - alpha)

    # Aplicar el umbral a los errores de reconstrucción del conjunto de prueba
    anomalies_conformal = np.where(test_recon_errors > quantile)[0]
    num_anomalies_conformal = len(anomalies_conformal)
    return anomalies_conformal, num_anomalies_conformal, quantile


# Predicción conformal con el Autoencoder Tradicional
trad_anomalies_conformal, trad_num_anomalies_conformal, trad_quantile = (
    conformal_prediction(traditional_model, train_loader, trad_recon_errors)
)
print(
    f"Número de anomalías detectadas con el Autoencoder Tradicional usando predicción conformal: {trad_num_anomalies_conformal}"
)

# Predicción conformal con el Autoencoder KAN
kan_anomalies_conformal, kan_num_anomalies_conformal, kan_quantile = (
    conformal_prediction(kan_model, train_loader, kan_recon_errors)
)
print(
    f"Número de anomalías detectadas con el Autoencoder KAN usando predicción conformal: {kan_num_anomalies_conformal}"
)

