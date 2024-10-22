import torch
import numpy as np
import random
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
from conformal_detection import detect_conformal_anomalies
from autoencoder_lstm import AutoencoderLSTM
from autoencoder_mlp import AutoencoderMLP
import matplotlib.pyplot as plt
import os
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # en caso de que no jale el gpu
    device = "cpu"

    print(f"Usando el dispositivo: {device}")

    file_path = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/Aventa_Taggenberg_16_02_2022.hdf5"
    json_file = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/Aventa_sensors.json"
    dataset_name = "Aventa"
    train_timestamp = "04_15_58"
    test_timestamps = ["17_12_32", "00_55_38", "04_15_58"]

    reader = HDF5Reader(file_path, json_file)
    reader.print_timestamps(dataset_name)

    df_train = reader.load_all_signals_for_timestamp(dataset_name, train_timestamp)
    # solo prueba
    #df_train = df_train.iloc[:1000]
    print(df_train.head())

    # Convertir los datos en secuencias para LSTM
    sequence_length = 10  # Puedes ajustar este valor según tus datos
    train_data = df_train.to_numpy()
    train_sequences = []
    for i in range(len(train_data) - sequence_length + 1):
        train_sequences.append(train_data[i:i+sequence_length])
    train_sequences = np.array(train_sequences)

    # Definir el modelo AutoencoderLSTM
    input_size = df_train.shape[1]
    hidden_size = 64  # Puedes ajustar este valor
    num_layers = 2    # Puedes ajustar este valor
    #model = AutoencoderLSTM(input_size, hidden_size, num_layers)
    model = AutoencoderMLP(input_size)

    # Determinar el tipo de modelo
    model_type = "lstm" if isinstance(model, AutoencoderLSTM) else "mpl"

    # Entrenar el modelo
    trained_model, loss_history = train_autoencoder(
        model, train_sequences, epochs=50, learning_rate=0.001, batch_size=16, device=device
    )

    os.makedirs('output', exist_ok=True)

    plt.figure()
    plt.plot(loss_history)
    plt.title('Función de pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.savefig(f'output/loss_plot_{model_type}.png')
    plt.close()

    # Guardar el historial de pérdidas con el sufijo del modelo
    pd.DataFrame(loss_history, columns=['Loss']).to_csv(f'output/loss_history_{model_type}.csv', index=False)

    for test_timestamp in test_timestamps:
        df_test = reader.load_all_signals_for_timestamp(dataset_name, test_timestamp)

        # Convertir los datos de prueba en secuencias
        test_data = df_test.to_numpy()
        test_sequences = []
        for i in range(len(test_data) - sequence_length + 1):
            test_sequences.append(test_data[i:i+sequence_length])
        test_sequences = np.array(test_sequences)

        reconstruction_error, state_labels, q_hat_prev, q_hat_total = detect_conformal_anomalies(
            trained_model,
            test_sequences,
            significance_level_prev=0.1,
            significance_level_total=0.05,
            device=device,
        )

        plt.figure()
        plt.plot(reconstruction_error, label='Error de reconstrucción')
        plt.axhline(y=q_hat_prev, color='r', linestyle='--', label='Umbral de fallo previo')
        plt.axhline(y=q_hat_total, color='g', linestyle='--', label='Umbral de fallo total')
        plt.title(f'Puntaje de anomalía para {test_timestamp}')
        plt.xlabel('Índice de muestra')
        plt.ylabel('Error de reconstrucción')
        plt.legend()
        plt.savefig(f'output/anomaly_score_{test_timestamp}_{model_type}.png')
        plt.close()

        reconstruction_error = np.array(reconstruction_error).flatten()
        state_labels = np.array(state_labels).flatten()

        # Guardar los resultados de reconstrucción con el sufijo del modelo
        pd.DataFrame({
            'Reconstruction Error': reconstruction_error,
            'State Label': state_labels
        }).to_csv(f'output/reconstruction_error_{test_timestamp}_{model_type}.csv', index=False)

if __name__ == "__main__":
    main()
