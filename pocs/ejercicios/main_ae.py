import sys
import os
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from lectura_archivos import h5_tree_to_dict
import seaborn as sns

# Configuración de rutas y semillas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('KAN_VAE_MNIST/efficient-kan-master')

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def leer_h5(file):
    with h5py.File(f"{file}.hdf5", 'r') as hf:
        h5_tree_dict = h5_tree_to_dict(hf)
    with open(f"{file}_structure_metadata.json", "w") as dump_file:
        json.dump(h5_tree_dict, dump_file)
    return h5_tree_dict

def leer_json(file):
    with open(file, "r") as tfile:
        return json.load(tfile)

def cargar_datasets(archivo_hdf5):
    datasets = {}
    with h5py.File(archivo_hdf5, "r") as f:
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

def procesar_datos_acelerometro(datasets, time_stamp, sensors_specs):
    df_all = pd.DataFrame()
    
    for sensor in sensors_specs['sensors']:
        try:
            ypr = sensor['sensor_placement'].get('yaw-pitch-roll', [0, 0, 0])
            rotation = Rotation.from_euler('ZYX', ypr, degrees=True)
        except (KeyError, ValueError) as e:
            print(f"Error al obtener o procesar yaw-pitch-roll para el sensor: {e}")
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
        
        if len(channel_data) == 3:  # Aseguramos que tenemos datos XYZ
            xyz_data = np.column_stack(channel_data)
            rotated_data = rotation.apply(xyz_data)
            
            for i, channel in enumerate(valid_channels):
                df_all[f"{channel}_Xt"] = rotated_data[:, 0]
                df_all[f"{channel}_Yt"] = rotated_data[:, 1]
                df_all[f"{channel}_Zt"] = rotated_data[:, 2]
    
    return df_all

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.reshape((x.shape[0], self.seq_len, self.n_features))
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        output, (hidden, cell) = self.decoder(hidden)
        return output

class SensorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.X = torch.tensor(data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx]

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
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, num_epochs=100):
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch in train_loader:
            optimizer.zero_grad()
            X_pred = model(X_batch)
            loss = criterion(X_pred, X_batch.reshape(X_pred.shape))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        history["train_loss"].append(train_loss)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch in val_loader:
                X_pred = model(X_batch)
                loss = criterion(X_pred, X_batch.reshape(X_pred.shape))
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Pérdida de Entrenamiento: {train_loss:.6f}, Pérdida de Validación: {val_loss:.6f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Detención temprana en la época {epoch+1}")
            break
    return history

def get_reconstruction_errors(model, loader):
    errors = []
    model.eval()
    with torch.no_grad():
        for X_batch in loader:
            X_pred = model(X_batch)
            loss = nn.functional.mse_loss(X_pred, X_batch.reshape(X_pred.shape), reduction='none')
            loss = loss.mean(dim=(1,2))
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def plot_error_distribution(train_errors, val_errors, test_errors, threshold, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='Entrenamiento')
    plt.hist(val_errors, bins=50, alpha=0.5, label='Validación')
    plt.hist(test_errors, bins=50, alpha=0.5, label='Prueba')
    plt.axvline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Error de reconstrucción')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de errores de reconstrucción')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(train_errors, test_errors, save_path):
    fpr, tpr, _ = roc_curve([0] * len(train_errors) + [1] * len(test_errors),
                            np.concatenate([train_errors, test_errors]))
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_error_time_series(test_errors, threshold, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(test_errors, label='Error de reconstrucción')
    plt.axhline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Muestra')
    plt.ylabel('Error de reconstrucción')
    plt.title('Serie temporal de errores de reconstrucción (Datos de prueba)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.title('Matriz de Confusión')
    plt.savefig(save_path)
    plt.close()

def plot_error_histogram(errors, threshold, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label='Umbral CP')
    plt.xlabel('Error de reconstrucción')
    plt.ylabel('Densidad')
    plt.title('Histograma de errores de reconstrucción')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(errors, thresholds, save_path):
    coverage = [np.mean(errors <= t) for t in thresholds]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, coverage)
    plt.xlabel('Umbral')
    plt.ylabel('Cobertura empírica')
    plt.title('Curva de calibración')
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed()
    
    # Crear carpeta de resultados si no existe
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Direcciones de archivos
    direccion_entrenamiento = "../aventa_normal_operation_for_system_identification/"
    direccion_test = "../aventa_rotor_icing/"
    
    # Cargar y procesar datos de entrenamiento
    file_entrenamiento = direccion_entrenamiento + "Aventa_Taggenberg_01_11_2022"
    datasets_entrenamiento = cargar_datasets(f"{file_entrenamiento}.hdf5")
    sensors_specs = leer_json(direccion_entrenamiento + "Aventa_sensors.json")
    
    df_all_entrenamiento = procesar_datos_acelerometro(datasets_entrenamiento, '11_06_44', sensors_specs)
    print("Datos de entrenamiento:")
    print(df_all_entrenamiento.columns)
    print(df_all_entrenamiento.head())

    # Cargar y procesar datos de prueba
    file_test = direccion_test + "Aventa_Taggenberg_01_11_2022"
    datasets_test = cargar_datasets(f"{file_test}.hdf5")
    
    df_all_test = procesar_datos_acelerometro(datasets_test, '11_06_44', sensors_specs)
    print("\nDatos de prueba:")
    print(df_all_test.columns)
    print(df_all_test.head())

    # Borrar en caso de querer senales reales
    df_all_entrenamiento = pd.read_csv('normal_signal.csv')
    df_all_test = pd.read_csv('anomalous_signal.csv')

    # Preparar datos para el autoencoder
    scaler = MinMaxScaler()
    df_scaled_entrenamiento = pd.DataFrame(scaler.fit_transform(df_all_entrenamiento.drop('Time', axis=1)), columns=df_all_entrenamiento.columns[1:])
    df_scaled_test = pd.DataFrame(scaler.transform(df_all_test.drop('Time', axis=1)), columns=df_all_test.columns[1:])
    
    dataset_entrenamiento = SensorDataset(df_scaled_entrenamiento)
    dataset_test = SensorDataset(df_scaled_test)

    # Dividir el dataset de entrenamiento
    train_size = int(0.8 * len(dataset_entrenamiento))
    val_size = len(dataset_entrenamiento) - train_size

    train_dataset, val_dataset = random_split(
        dataset_entrenamiento,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Crear DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Configurar y entrenar el modelo
    seq_len = 1
    n_features = df_scaled_entrenamiento.shape[1]
    embedding_dim = 64

    model = LSTMAutoencoder(seq_len, n_features, embedding_dim)

    start_time = time.time()
    history = train_model(model, train_loader, val_loader, num_epochs=100)
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")

    # Evaluar el modelo y aplicar Conformal Prediction
    train_errors = get_reconstruction_errors(model, train_loader)
    val_errors = get_reconstruction_errors(model, val_loader)
    test_errors = get_reconstruction_errors(model, test_loader)

    # Calcular umbral de Conformal Prediction
    alpha = 0.05  # Nivel de significancia
    calibration_errors = np.concatenate([train_errors, val_errors])
    threshold = np.percentile(calibration_errors, (1 - alpha) * 100)

    # Detectar anomalías
    test_predictions = (test_errors > threshold).astype(int)

    # Generar y guardar gráficos
    plot_error_distribution(train_errors, val_errors, test_errors, threshold, os.path.join(results_dir, 'ae_distribucion_errores.png'))
    plot_roc_curve(train_errors, test_errors, os.path.join(results_dir, 'ae_curva_roc.png'))
    plot_error_time_series(test_errors, threshold, os.path.join(results_dir, 'ae_serie_temporal_errores.png'))
    plot_confusion_matrix([0] * len(train_errors) + [1] * len(test_errors), 
                          np.concatenate([train_errors > threshold, test_errors > threshold]),
                          os.path.join(results_dir, 'ae_matriz_confusion.png'))
    plot_error_histogram(test_errors, threshold, os.path.join(results_dir, 'ae_histograma_errores.png'))
    
    thresholds = np.linspace(np.min(calibration_errors), np.max(calibration_errors), 100)
    plot_calibration_curve(calibration_errors, thresholds, os.path.join(results_dir, 'ae_curva_calibracion.png'))

    # Calcular métricas adicionales
    precision = precision_score([0] * len(train_errors) + [1] * len(test_errors), 
                                np.concatenate([train_errors > threshold, test_errors > threshold]))
    recall = recall_score([0] * len(train_errors) + [1] * len(test_errors), 
                          np.concatenate([train_errors > threshold, test_errors > threshold]))
    f1 = f1_score([0] * len(train_errors) + [1] * len(test_errors), 
                  np.concatenate([train_errors > threshold, test_errors > threshold]))

    # Imprimir estadísticas
    print(f"\nUmbral de Conformal Prediction: {threshold:.4f}")
    print(f"Porcentaje de anomalías detectadas: {np.mean(test_predictions) * 100:.2f}%")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Guardar resultados
    results_df = pd.DataFrame({
        'Error_reconstruccion': test_errors,
        'Es_anomalia': test_predictions
    })
    results_df.to_csv(os.path.join(results_dir, 'ae_resultados_anomalias.csv'), index=False)

    print(f"\nResultados guardados en '{results_dir}/ae_resultados_anomalias.csv'")
    print(f"Gráficos guardados en la carpeta '{results_dir}'")

if __name__ == "__main__":
    main()