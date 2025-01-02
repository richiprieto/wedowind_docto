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


def set_seed(seed):
    """Establece la semilla para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Obtiene el dispositivo (CPU o GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    return device


def load_data(num_samples, seed):
    """Crea un DataFrame de ejemplo."""
    np.random.seed(seed)
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
    return df


def preprocess_data(df):
    """Preprocesa y normaliza los datos."""
    features = df.drop("time", axis=1)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


def split_data(X, seed):
    """Divide los datos en entrenamiento, validación y prueba."""
    X = torch.tensor(X, dtype=torch.float32)
    X_temp, X_test = train_test_split(X.cpu().numpy(), test_size=0.2, random_state=seed)
    X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=seed)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    return X_train, X_val, X_test


class TraditionalAutoencoder(nn.Module):
    """Clase para el autoencoder tradicional."""

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


def initialize_kan(input_dim, seed, device):
    """Inicializa el modelo KAN."""
    model = KAN(width=[input_dim, 16, input_dim], grid=5, k=3, seed=seed, device=device)
    torch.set_default_dtype(torch.float32)
    return model


def train_autoencoder(
    model, train_loader, val_loader, device, num_epochs=100, patience=10
):
    """Entrena el autoencoder tradicional con early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
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
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                # Cargar los mejores pesos
                model.load_state_dict(best_model_wts)
                break
    return model


def evaluate_model(model, test_loader, device, alpha=0.05):
    """Evalúa el autoencoder tradicional y detecta anomalías."""
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.extend(loss.cpu().numpy())

    # Establecer un umbral para la detección de anomalías
    threshold = np.percentile(reconstruction_errors, 100 * (1 - alpha))

    # Detectar anomalías
    anomalies = np.where(reconstruction_errors > threshold)[0]
    print(
        f"Número de anomalías detectadas con el Autoencoder Tradicional: {len(anomalies)}"
    )

    return reconstruction_errors, anomalies, threshold


def conformal_prediction(errors, calibration_errors, alpha=0.05):
    """Realiza la predicción conformal para detección de anomalías."""
    quantile = np.quantile(calibration_errors, 1 - alpha)
    anomalies_conformal = np.where(errors > quantile)[0]
    print(
        f"Número de anomalías detectadas usando predicción conformal: {len(anomalies_conformal)}"
    )
    return anomalies_conformal, quantile


def train_kan(model, dataset, num_epochs=100, patience=10):
    """Entrena el modelo KAN con early stopping."""
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.fit(dataset, opt="LBFGS", steps=1, lamb=0.001)

        # Validación
        with torch.no_grad():
            outputs = model(dataset["test_input"])
            val_loss = torch.mean((outputs - dataset["test_label"]) ** 2).item()

        print(f"Epoch {epoch+1}/{num_epochs}, KAN Validation Loss: {val_loss:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Guardar los mejores pesos
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping para KAN!")
                # Cargar los mejores pesos
                model.load_state_dict(best_model_wts)
                break
    return model


def evaluate_kan(model, dataset, alpha=0.05):
    """Evalúa el modelo KAN y detecta anomalías."""
    with torch.no_grad():
        outputs = model(dataset["test_input"])
        reconstruction_errors = (
            torch.mean((outputs - dataset["test_label"]) ** 2, dim=1).cpu().numpy()
        )

    # Establecer un umbral para la detección de anomalías
    threshold = np.percentile(reconstruction_errors, 100 * (1 - alpha))

    # Detectar anomalías
    anomalies = np.where(reconstruction_errors > threshold)[0]
    print(f"Número de anomalías detectadas con KAN: {len(anomalies)}")

    return reconstruction_errors, anomalies, threshold


def main():
    # Configuración inicial
    seed = 42
    set_seed(seed)
    device = get_device()

    # Carga y preprocesamiento de datos
    num_samples = 10000
    df = load_data(num_samples, seed)
    features_scaled, scaler = preprocess_data(df)
    X_train, X_val, X_test = split_data(features_scaled, seed)

    # Mover datos al dispositivo
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    # Crear datasets y dataloaders para el autoencoder tradicional
    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    # Definir y entrenar el autoencoder tradicional
    traditional_model = TraditionalAutoencoder(input_dim).to(device)
    traditional_model = train_autoencoder(
        traditional_model, train_loader, val_loader, device
    )

    # Evaluación y detección de anomalías con el autoencoder tradicional
    reconstruction_errors, anomalies, threshold = evaluate_model(
        traditional_model, test_loader, device
    )

    # Predicción conformal para el autoencoder tradicional
    calibration_errors = []
    traditional_model.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = traditional_model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=1)
            calibration_errors.extend(loss.cpu().numpy())
    anomalies_conformal, quantile = conformal_prediction(
        reconstruction_errors, calibration_errors
    )

    # Preparar dataset para KAN
    dataset = {
        "train_input": X_train,
        "train_label": X_train,
        "test_input": X_test,
        "test_label": X_test,
    }

    # Definir y entrenar el modelo KAN
    kan_model = initialize_kan(input_dim, seed, device)
    kan_model = train_kan(kan_model, dataset)

    # Evaluación y detección de anomalías con KAN
    reconstruction_errors_kan, anomalies_kan, threshold_kan = evaluate_kan(
        kan_model, dataset
    )

    # Predicción conformal para KAN
    with torch.no_grad():
        outputs_train = kan_model(dataset["train_input"])
        calibration_errors_kan = (
            torch.mean((outputs_train - dataset["train_label"]) ** 2, dim=1)
            .cpu()
            .numpy()
        )
    anomalies_conformal_kan, quantile_kan = conformal_prediction(
        reconstruction_errors_kan, calibration_errors_kan
    )

    # Opcional: Mostrar fórmula simbólica de KAN
    # lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
    # kan_model.auto_symbolic(lib=lib)
    # formula = ex_round(kan_model.symbolic_formula()[0][0], 4)
    # print(f'Fórmula simbólica aproximada: {formula}')


if __name__ == "__main__":
    main()
