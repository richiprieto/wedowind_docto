import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from typing import Dict, List, Tuple, Union
import h5py
import json
from scipy.spatial.transform import Rotation
from scipy.stats import norm
# Configuración
SEED = 42
BATCH_SIZE = 128
EMBEDDING_DIM = 64
NUM_EPOCHS = 100
RESULTS_DIR = "results"

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = EMBEDDING_DIM):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(decoder_input)
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]

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
            X_pred = model(X_batch)
            loss = nn.functional.mse_loss(X_pred, X_batch, reduction='none').mean(dim=(1,2))
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

class ConformalAnomalyDetector:
    def __init__(self, calibration_errors: np.ndarray, significance: float = 0.05):
        self.calibration_errors = calibration_errors
        self.significance = significance
        self.threshold = np.percentile(calibration_errors, (1 - significance) * 100)

    def detect_anomalies(self, test_errors: np.ndarray) -> np.ndarray:
        return test_errors > self.threshold

    def set_significance(self, significance: float):
        self.significance = significance
        self.threshold = np.percentile(self.calibration_errors, (1 - significance) * 100)

    def evaluate(self, test_errors: np.ndarray, true_anomalies: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        predicted_anomalies = self.detect_anomalies(test_errors)
        
        accuracy = accuracy_score(true_anomalies, predicted_anomalies)
        precision = precision_score(true_anomalies, predicted_anomalies, zero_division=0)
        recall = recall_score(true_anomalies, predicted_anomalies, zero_division=0)
        f1 = f1_score(true_anomalies, predicted_anomalies, zero_division=0)
        
        if len(np.unique(true_anomalies)) > 1:
            auc_roc = roc_auc_score(true_anomalies, predicted_anomalies)
            fpr, tpr, _ = roc_curve(true_anomalies, predicted_anomalies)
        else:
            auc_roc = None
            fpr, tpr = None, None
        
        cm = confusion_matrix(true_anomalies, predicted_anomalies)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        }

def plot_results(train_errors: np.ndarray, val_errors: np.ndarray, test_errors: np.ndarray, 
                 threshold: float, predictions: np.ndarray, significance: float):
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
    plt.savefig(os.path.join(RESULTS_DIR, f'error_distribution_sig_{significance}.png'))
    plt.close()
    
    # Curva ROC
    try:
        fpr, tpr, _ = roc_curve(np.concatenate([np.zeros_like(train_errors), np.ones_like(test_errors)]),
                                np.concatenate([train_errors, test_errors]))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Significance = {significance})')
        plt.legend(loc="lower right")
    except ValueError:
        print(f"No se pudo calcular la curva ROC para significancia {significance}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No se pudo calcular la curva ROC\nTodos los valores son de la misma clase", 
                 ha='center', va='center', fontsize=12)
        plt.title(f'Curva ROC no disponible (Significancia = {significance})')

    plt.savefig(os.path.join(RESULTS_DIR, f'roc_curve_sig_{significance}.png'))
    plt.close()
    
    # Serie temporal de errores
    plt.figure(figsize=(12, 6))
    plt.plot(test_errors, label='Reconstruction Error')
    plt.axhline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Sample')
    plt.ylabel('Reconstruction Error')
    plt.title(f'Time Series of Reconstruction Errors (Significance = {significance})')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'error_time_series_sig_{significance}.png'))
    plt.close()
    
    # Matriz de confusión
    cm = confusion_matrix(np.concatenate([np.zeros_like(train_errors), np.ones_like(test_errors)]), 
                          np.concatenate([train_errors > threshold, predictions]))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Significance = {significance})')
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_sig_{significance}.png'))
    plt.close()

    # 1. Histograma de la Medida de No Conformidad
    plt.figure(figsize=(10, 6))
    sns.histplot(data=test_errors, kde=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Medida de No Conformidad')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la Medida de No Conformidad')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'histograma_no_conformidad_sig_{significance}.png'))
    plt.close()

    # 2. Distribución de p-valores
    p_values = 1 - norm.cdf(test_errors)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=p_values, kde=True)
    plt.axvline(significance, color='r', linestyle='--', label='Nivel de significancia')
    plt.xlabel('p-valor')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de p-valores')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'distribucion_p_valores_sig_{significance}.png'))
    plt.close()

    # 3. Gráfico de Dispersión de Medida de No Conformidad
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(test_errors)), test_errors, c=predictions, cmap='coolwarm')
    plt.axhline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Medida de No Conformidad')
    plt.title('Gráfico de Dispersión de Medida de No Conformidad')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'dispersion_no_conformidad_sig_{significance}.png'))
    plt.close()

    # 4. Gráfico de Línea del Valor de No Conformidad Ordenado
    sorted_errors = np.sort(test_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_errors)), sorted_errors)
    plt.axhline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Índice de muestra ordenado')
    plt.ylabel('Medida de No Conformidad')
    plt.title('Valor de No Conformidad Ordenado')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'no_conformidad_ordenado_sig_{significance}.png'))
    plt.close()

    # 5. Gráfico de Anomalías Detectadas
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(test_errors)), test_errors, c=predictions, cmap='coolwarm')
    plt.axhline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Medida de No Conformidad')
    plt.title('Anomalías Detectadas')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'anomalias_detectadas_sig_{significance}.png'))
    plt.close()

    # 6. Gráfico de Series Temporales con Anomalías
    plt.figure(figsize=(12, 6))
    plt.plot(test_errors, label='Medida de No Conformidad')
    plt.scatter(range(len(test_errors)), test_errors, c=predictions, cmap='coolwarm')
    plt.axhline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Tiempo')
    plt.ylabel('Medida de No Conformidad')
    plt.title('Series Temporales con Anomalías')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'series_temporales_anomalias_sig_{significance}.png'))
    plt.close()

    # 7. Mapa de Calor (Heatmap)
    plt.figure(figsize=(12, 8))
    # Calculamos las dimensiones para el reshape
    num_rows = int(np.sqrt(len(test_errors)))
    num_cols = int(np.ceil(len(test_errors) / num_rows))
    
    # Reshape y rellena con NaN si es necesario
    heatmap_data = np.pad(test_errors, (0, num_rows * num_cols - len(test_errors)), mode='constant', constant_values=np.nan)
    heatmap_data = heatmap_data.reshape(num_rows, num_cols)
    
    sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Medida de No Conformidad'})
    plt.xlabel('Columna')
    plt.ylabel('Fila')
    plt.title('Mapa de Calor de Medidas de No Conformidad')
    plt.savefig(os.path.join(RESULTS_DIR, f'heatmap_no_conformidad_sig_{significance}.png'))
    plt.close()


def plot_metrics_summary(metrics_list, significances):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    data = {metric: [m[metric] if m[metric] is not None else 0 for m in metrics_list] for metric in metrics}
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(significances))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, data[metric], width, label=metric)
    
    plt.xlabel('Significance Level')
    plt.ylabel('Metric Value')
    plt.title('Summary of Metrics for Different Significance Levels')
    plt.xticks(x + width*2, significances)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_summary.png'))
    plt.close()

def plot_loss_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_history.png'))
    plt.close()

def save_results(results: pd.DataFrame, file_name: str):
    results.to_csv(os.path.join(RESULTS_DIR, file_name), index=False)
    print(f"Results saved in {os.path.join(RESULTS_DIR, file_name)}")

def load_datasets(file_path: str) -> Dict:
    datasets = {}
    with h5py.File(file_path, "r") as f:
        for dataset_name in f.keys():
            datasets[dataset_name] = {}
            for time_stamp in f[dataset_name].keys():
                datasets[dataset_name][time_stamp] = {}
                signal_list = list(f[dataset_name][time_stamp].keys())
                signal_list.remove('ChannelList')
                for signal in signal_list:
                    datasets[dataset_name][time_stamp][signal] = {
                        'Time': f[dataset_name][time_stamp][signal]['Time'][()],
                        'Value': f[dataset_name][time_stamp][signal]['Value'][()]
                    }
    return datasets

def process_accelerometer_data(datasets: Dict, time_stamp: str, sensors_specs: Dict) -> pd.DataFrame:
    df_all = pd.DataFrame()
    
    for sensor in sensors_specs['sensors']:
        try:
            ypr = sensor['sensor_placement'].get('yaw-pitch-roll', [0, 0, 0])
            rotation = Rotation.from_euler('ZYX', ypr, degrees=True)
        except (KeyError, ValueError) as e:
            print(f"Error processing yaw-pitch-roll for sensor: {e}")
            rotation = Rotation.from_euler('ZYX', [0, 0, 0], degrees=True)
        
        channel_data = []
        valid_channels = []
        for channel in sensor['channels']:
            if channel in datasets['Aventa'][time_stamp]:
                data = datasets['Aventa'][time_stamp][channel]['Value'].flatten()
                channel_data.append(data)
                valid_channels.append(channel)
                if 'Time' not in df_all:
                    df_all['Time'] = datasets['Aventa'][time_stamp][channel]['Time'].flatten()
        
        if len(channel_data) == 3:  # Ensure we have XYZ data
            xyz_data = np.column_stack(channel_data)
            rotated_data = rotation.apply(xyz_data)
            
            for i, channel in enumerate(valid_channels):
                df_all[f"{channel}_Xt"] = rotated_data[:, 0]
                df_all[f"{channel}_Yt"] = rotated_data[:, 1]
                df_all[f"{channel}_Zt"] = rotated_data[:, 2]
    
    return df_all

def main():
    set_seed()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # File paths
    train_dir = "../aventa_normal_operation_for_system_identification/"
    test_dir = "../aventa_rotor_icing/"
    #test_dir = train_dir
    
    # Load and process training data
    train_file = train_dir + "Aventa_Taggenberg_01_11_2022.hdf5"
    train_datasets = load_datasets(train_file)
    with open(train_dir + "Aventa_sensors.json", "r") as f:
        sensors_specs = json.load(f)
    
    df_train = process_accelerometer_data(train_datasets, '12_52_05', sensors_specs)
    
    # Load and process test data
    test_file = test_dir + "Aventa_Taggenberg_01_11_2022.hdf5"
    test_datasets = load_datasets(test_file)
    
    df_test = process_accelerometer_data(test_datasets, '12_52_05', sensors_specs)
    
    # Uncomment these lines if you want to use the CSV files instead
    #df_train = pd.read_csv('normal_signal.csv')
    #df_test = pd.read_csv('anomalous_signal.csv')
    
    # Prepare data for the autoencoder
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.drop('Time', axis=1))
    X_test = scaler.transform(df_test.drop('Time', axis=1))
    
    # Configure parameters
    seq_len = 10  # Sequence length for LSTM
    n_features = X_train.shape[1]
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, seq_len)
    test_dataset = TimeSeriesDataset(X_test, seq_len)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Configure and train the model
    model = LSTMAutoencoder(seq_len, n_features, EMBEDDING_DIM)
    history = train_model(model, train_loader, val_loader)
    
    # Plot loss history
    plot_loss_history(history)
    
    # Evaluate the model and apply Conformal Prediction
    train_errors = get_reconstruction_errors(model, train_loader)
    val_errors = get_reconstruction_errors(model, val_loader)
    test_errors = get_reconstruction_errors(model, test_loader)
    
    calibration_errors = np.concatenate([train_errors, val_errors])
    
    significances = [0.025, 0.05, 0.25, 0.5]
    all_metrics = []
    all_results = []
    
    for significance in significances:
        conformal_detector = ConformalAnomalyDetector(calibration_errors, significance)
        predictions = conformal_detector.detect_anomalies(test_errors)
        
        # Assume all samples in the test set are anomalous
        true_anomalies = np.ones_like(test_errors, dtype=bool)
        
        metrics = conformal_detector.evaluate(test_errors, true_anomalies)
        all_metrics.append(metrics)
        
        print(f"\nMetrics for significance level {significance}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Generate and save plots
        plot_results(train_errors, val_errors, test_errors, conformal_detector.threshold, predictions, significance)
        
        # Save results
        results = {
            "reconstruction_error": test_errors,
            "is_anomaly": predictions.astype(int),
            "threshold": conformal_detector.threshold,
            "significance": significance,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        all_results.append(results)
    
    # Generate summary plot of metrics
    plot_metrics_summary(all_metrics, significances)
    
    # Save all results
    combined_results = pd.concat([pd.DataFrame(r) for r in all_results])
    save_results(combined_results, "anomaly_detection_results.csv")
    
    print(f"\nResults and plots saved in the '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main() 