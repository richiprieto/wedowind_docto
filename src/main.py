import torch
import numpy as np
import random
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
from conformal_detection import detect_conformal_anomalies
from autoencoder_mlp import AutoencoderMLP
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    file_path = "../../dataset/Aventa_Taggenberg_16_02_2022.hdf5"
    json_file = "../../dataset/Aventa_sensors.json"
    dataset_name = "Aventa"
    train_timestamp = "04_15_58"
    test_timestamps = ["17_12_32", "00_55_38", "04_15_58"]

    reader = HDF5Reader(file_path, json_file)
    reader.print_timestamps(dataset_name)

    df_train = reader.load_all_signals_for_timestamp(dataset_name, train_timestamp)
    df_train = df_train.drop(columns=['Time'])  # Eliminar la columna 'Time'
    print(df_train.head())

    # Normalizar el dataset de entrenamiento
    scaler = StandardScaler()
    df_train_normalized = scaler.fit_transform(df_train)

    # Train with the entire dataset
    model = AutoencoderMLP(input_size=df_train.shape[1])
    trained_model, loss_history = train_autoencoder(
        model, df_train_normalized, epochs=50, learning_rate=0.001, batch_size=32, device=device
    )

    os.makedirs('output', exist_ok=True)

    plt.figure()
    plt.plot(loss_history)
    plt.title('Función de pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.savefig('output/loss_plot.png')
    plt.close()

    pd.DataFrame(loss_history, columns=['Loss']).to_csv('output/loss_history.csv', index=False)

    for test_timestamp in test_timestamps:
        df_test = reader.load_all_signals_for_timestamp(dataset_name, test_timestamp)
        df_test = df_test.drop(columns=['Time'])  # Eliminar la columna 'Time'

        # Normalizar el dataset de prueba
        df_test_normalized = scaler.transform(df_test)

        reconstruction_error, state_labels, q_hat_prev, q_hat_total = (
            detect_conformal_anomalies(
                trained_model,
                df_test_normalized,
                significance_level_prev=0.1,
                significance_level_total=0.05,
                device=device,
            )
        )

        plt.figure()
        plt.plot(reconstruction_error, label='Error de reconstrucción')
        plt.axhline(y=q_hat_prev, color='r', linestyle='--', label='Umbral de fallo previo')
        plt.axhline(y=q_hat_total, color='g', linestyle='--', label='Umbral de fallo total')
        plt.title(f'Puntaje de anomalía para {test_timestamp}')
        plt.xlabel('Índice de muestra')
        plt.ylabel('Error de reconstrucción')
        plt.legend()
        plt.savefig(f'output/anomaly_score_{test_timestamp}.png')
        plt.close()

        reconstruction_error = np.array(reconstruction_error).flatten()
        state_labels = np.array(state_labels).flatten()

        pd.DataFrame({
            'Reconstruction Error': reconstruction_error,
            'State Label': state_labels
        }).to_csv(f'output/reconstruction_error_{test_timestamp}.csv', index=False)

if __name__ == "__main__":
    main()
