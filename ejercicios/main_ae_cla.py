import sys
import os
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Union


# Configuración
SEED = 42
BATCH_SIZE = 128
EMBEDDING_DIM = 64
NUM_EPOCHS = 100
RESULTS_DIR = "results"

def set_seed(seed: int = SEED):
    """Configura las semillas para reproducibilidad."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_h5_data(file_path: str) -> Dict:
    """Carga datos de un archivo HDF5."""
    with h5py.File(file_path, 'r') as f:
        return {dataset: {time_stamp: {signal: {
            'Time': f[dataset][time_stamp][signal]['Time'][()],
            'Value': f[dataset][time_stamp][signal]['Value'][()]
        } for signal in f[dataset][time_stamp].keys() if signal != 'ChannelList'}
        for time_stamp in f[dataset].keys()} for dataset in f.keys()}

def load_json(file_path: str) -> Dict:
    """Carga datos de un archivo JSON."""
    with open(file_path, 'r') as file:
        return json.load(file)

def process_accelerometer_data(datasets: Dict, time_stamp: str, sensors_specs: Dict) -> pd.DataFrame:
    """Procesa los datos del acelerómetro aplicando rotaciones."""
    df_all = pd.DataFrame()
    
    for sensor in sensors_specs['sensors']:
        ypr = sensor['sensor_placement'].get('yaw-pitch-roll', [0, 0, 0])
        rotation = Rotation.from_euler('ZYX', ypr, degrees=True)
        
        channel_data = []
        valid_channels = []
        for channel in sensor['channels']:
            if channel in datasets['Aventa'][time_stamp]:
                data = datasets['Aventa'][time_stamp][channel]['Value'].flatten()
                channel_data.append(data)
                valid_channels.append(channel)
                if 'Time' not in df_all:
                    df_all['Time'] = datasets['Aventa'][time_stamp][channel]['Time'].flatten()
        
        if len(channel_data) == 3:
            xyz_data = np.column_stack(channel_data)
            rotated_data = rotation.apply(xyz_data)
            
            for i, channel in enumerate(valid_channels):
                df_all[f"{channel}_Xt"] = rotated_data[:, 0]
                df_all[f"{channel}_Yt"] = rotated_data[:, 1]
                df_all[f"{channel}_Zt"] = rotated_data[:, 2]
    
    return df_all

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = EMBEDDING_DIM):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        _, (hidden, _) = self.encoder(x)
        # hidden shape: (1, batch_size, embedding_dim)
        
        # Repetir el estado oculto para cada paso de tiempo
        decoder_input = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        # decoder_input shape: (batch_size, seq_len, embedding_dim)
        
        output, _ = self.decoder(decoder_input)
        # output shape: (batch_size, seq_len, n_features)
        return output

class SensorDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = torch.tensor(data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = NUM_EPOCHS) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(patience=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch in train_loader:
            X_batch = X_batch.to(device)
            X_batch = X_batch.unsqueeze(1)  # Añadir dimensión de secuencia
            optimizer.zero_grad()
            X_pred = model(X_batch)
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch in val_loader:
                X_batch = X_batch.to(device)
                X_batch = X_batch.unsqueeze(1)  # Añadir dimensión de secuencia
                X_pred = model(X_batch)
                loss = criterion(X_pred, X_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history

def get_reconstruction_errors(model: nn.Module, loader: DataLoader) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            X_batch = X_batch.unsqueeze(1)  # Añadir dimensión de secuencia
            X_pred = model(X_batch)
            loss = nn.functional.mse_loss(X_pred, X_batch, reduction='none').mean(dim=(1,2))
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def conformal_prediction(calibration_errors: np.ndarray, test_errors: np.ndarray, alpha: float = 0.05) -> Tuple[float, np.ndarray]:
    """
    Implementa Conformal Prediction para detectar anomalías.
    
    Args:
    calibration_errors: Errores de reconstrucción del conjunto de calibración.
    test_errors: Errores de reconstrucción del conjunto de prueba.
    alpha: Nivel de significancia (por defecto 0.05 para un 95% de confianza).
    
    Returns:
    threshold: Umbral de anomalía.
    predictions: Array booleano indicando si cada muestra de prueba es una anomalía.
    """
    n_calibration = len(calibration_errors)
    augmented_errors = np.concatenate([calibration_errors, test_errors])
    scores = np.zeros(len(augmented_errors))
    
    for i, error in enumerate(augmented_errors):
        scores[i] = np.sum(augmented_errors <= error) / (n_calibration + 1)
    
    threshold = np.percentile(calibration_errors, (1 - alpha) * 100)
    predictions = test_errors > threshold
    
    return threshold, predictions

def plot_results(train_errors: np.ndarray, val_errors: np.ndarray, test_errors: np.ndarray, threshold: float, predictions: np.ndarray):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Distribución de errores
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='Train')
    plt.hist(val_errors, bins=50, alpha=0.5, label='Validation')
    plt.hist(test_errors, bins=50, alpha=0.5, label='Test')
    plt.axvline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_error_distribution.png'))
    plt.close()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve([0] * len(train_errors) + [1] * len(test_errors),
                            np.concatenate([train_errors, test_errors]))
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_roc_curve.png'))
    plt.close()
    
    # Serie temporal de errores
    plt.figure(figsize=(12, 6))
    plt.plot(test_errors, label='Reconstruction Error')
    plt.axhline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Sample')
    plt.ylabel('Reconstruction Error')
    plt.title('Time Series of Reconstruction Errors (Test Data)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_error_time_series.png'))
    plt.close()
    
    # Matriz de confusión
    cm = confusion_matrix([0] * len(train_errors) + [1] * len(test_errors), 
                          np.concatenate([train_errors > threshold, predictions]))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_confusion_matrix.png'))
    plt.close()
    
    # Histograma de errores
    plt.figure(figsize=(10, 6))
    plt.hist(test_errors, bins=50, density=True, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Histogram of Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_error_histogram.png'))
    plt.close()
    
    # Curva de calibración
    thresholds = np.linspace(np.min(val_errors), np.max(val_errors), 100)
    coverage = [np.mean(val_errors <= t) for t in thresholds]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, coverage)
    plt.xlabel('Threshold')
    plt.ylabel('Empirical Coverage')
    plt.title('Calibration Curve')
    plt.savefig(os.path.join(RESULTS_DIR, 'ae_calibration_curve.png'))
    plt.close()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento para la detección de anomalías.
    
    Args:
    y_true: Etiquetas verdaderas (0 para normal, 1 para anomalía).
    y_pred: Predicciones (0 para normal, 1 para anomalía).
    
    Returns:
    Dict con las métricas calculadas.
    """
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred)
    }

def save_results(results: Dict[str, Union[float, np.ndarray]], file_name: str):
    """
    Guarda los resultados en un archivo CSV.
    
    Args:
    results: Diccionario con los resultados a guardar.
    file_name: Nombre del archivo CSV.
    """
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, file_name), index=False)
    print(f"Resultados guardados en {os.path.join(RESULTS_DIR, file_name)}")


def main():
    set_seed()
    
    # Cargar y procesar datos
    train_file = "../aventa_normal_operation_for_system_identification/Aventa_Taggenberg_01_11_2022.hdf5"
    test_file = "../aventa_rotor_icing/Aventa_Taggenberg_01_11_2022.hdf5"
    sensors_file = "../aventa_normal_operation_for_system_identification/Aventa_sensors.json"
    
    train_data = load_h5_data(train_file)
    test_data = load_h5_data(test_file)
    sensors_specs = load_json(sensors_file)
    
    #df_train = process_accelerometer_data(train_data, '11_06_44', sensors_specs)
    #df_test = process_accelerometer_data(test_data, '11_06_44', sensors_specs)

    # Borrar en caso de querer senales reales
    df_train = pd.read_csv('normal_signal.csv')
    df_test = pd.read_csv('anomalous_signal.csv')
    
    # Preparar datos para el autoencoder
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(df_train.drop('Time', axis=1)), columns=df_train.columns[1:])
    X_test = pd.DataFrame(scaler.transform(df_test.drop('Time', axis=1)), columns=df_test.columns[1:])
    
    train_dataset = SensorDataset(X_train)
    test_dataset = SensorDataset(X_test)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Configurar y entrenar el modelo
    seq_len = 1
    n_features = X_train.shape[1]
    model = LSTMAutoencoder(seq_len, n_features, EMBEDDING_DIM)
    
    history = train_model(model, train_loader, val_loader)
    
    # Evaluar el modelo y aplicar Conformal Prediction
    train_errors = get_reconstruction_errors(model, train_loader)
    val_errors = get_reconstruction_errors(model, val_loader)
    test_errors = get_reconstruction_errors(model, test_loader)
    
    calibration_errors = np.concatenate([train_errors, val_errors])
    threshold, predictions = conformal_prediction(calibration_errors, test_errors)
    
    # Generar y guardar gráficos
    plot_results(train_errors, val_errors, test_errors, threshold, predictions)
    
    # Calcular métricas
    y_true = np.concatenate([np.zeros_like(train_errors), np.ones_like(test_errors)])
    y_pred = np.concatenate([train_errors > threshold, predictions])
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Guardar resultados
    results = {
        "reconstruction_error": test_errors,
        "is_anomaly": predictions.astype(int),
        "threshold": threshold,
        **metrics
    }
    save_results(results, "ae_anomaly_detection_results.csv")
    
    print(f"\nResults and plots saved in '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()    
