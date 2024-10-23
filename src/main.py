import torch
import numpy as np
import random
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
from conformal_detection import conformal_anomaly_detection
from autoencoder_mlp import AutoencoderMLP
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Configuración inicial
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    device = "cpu"

    # Cargar el dataset saludable
    path_saludable = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/"
    file_path_train = os.path.join(path_saludable, "Aventa_Taggenberg_06_02_2022.hdf5")
    json_file_train = os.path.join(path_saludable, "Aventa_sensors.json")
    dataset_name = "Aventa"

    reader_train = HDF5Reader(file_path_train, json_file_train)
    train_timestamps = reader_train.print_timestamps(dataset_name)
    df = reader_train.load_all_signals_for_timestamps(dataset_name, train_timestamps)
    df = df.drop(columns=['Time', 'Timestamp'])  # Eliminar columnas no necesarias

    # Dividir en entrenamiento, calibración y validación
    df_train_full, df_valid = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_calibration = train_test_split(df_train_full, test_size=0.25, random_state=42)
    # df_train: 60%, df_calibration: 20%, df_valid: 20%

    #df_train = df_train.drop(columns=['Time', 'Timestamp'])
    #df_calibration = df_calibration.drop(columns=['Time', 'Timestamp'])
    #df_valid = df_valid.drop(columns=['Time', 'Timestamp'])

    df_train = df_train.iloc[-240000:, :]

    print(df_train.head())

    # Normalizar los datasets
    scaler = MinMaxScaler()
    df_train_normalized = scaler.fit_transform(df_train)
    df_calibration_normalized = scaler.transform(df_calibration)
    df_valid_normalized = scaler.transform(df_valid)

    # Entrenar el autoencoder con el conjunto de entrenamiento
    print("Entrenando el autoencoder")
    input_size = df_train.shape[1]
    model = AutoencoderMLP(input_size).to(device)
    trained_model, loss_history = train_autoencoder(
        model, df_train_normalized, epochs=50, learning_rate=0.001, batch_size=32, device=device
    )

    # Guardar gráfico de pérdida
    os.makedirs('output', exist_ok=True)
    plt.figure()
    plt.plot(loss_history)
    plt.title('Función de pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.savefig('output/loss_plot.png')
    plt.close()

    # Detección de anomalías en el conjunto de validación
    print("Realizando detección de anomalías en el conjunto de validación")
    reconstruction_error, q_hat = conformal_anomaly_detection(
        trained_model,
        df_valid_normalized,
        calibration_data=df_calibration_normalized,
        significance_level=0.05,
        device=device
    )

    # Mostrar resultados
    plt.figure()
    plt.plot(reconstruction_error, label='Error de reconstrucción')
    plt.axhline(y=q_hat, color='r', linestyle='--', label='Umbral de anomalía')
    plt.title('Detección de anomalías en conjunto de validación')
    plt.xlabel('Muestras')
    plt.ylabel('Error de reconstrucción')
    plt.legend()
    plt.savefig('output/anomaly_detection.png')
    plt.close()

    print("Proceso completado. Los resultados se han guardado en la carpeta 'output'.")

if __name__ == "__main__":
    main()
