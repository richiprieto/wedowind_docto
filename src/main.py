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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse  # Importación añadida

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento y detección de anomalías con Autoencoder")
    parser.add_argument('--only_testing', action='store_true', help='Modo solo prueba: omite el entrenamiento y realiza la inferencia.')
    args = parser.parse_args()

    # Configuración inicial
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # en caso de que no jale el gpu
    device = "cpu"
    print(f"Usando el dispositivo: {device}")

    # Asegúrate de que el directorio 'output' existe
    os.makedirs('output', exist_ok=True)

    # Cargar el dataset saludable
    path_saludable = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/"
    file_path_train = os.path.join(path_saludable, "Aventa_Taggenberg_06_02_2022.hdf5")
    json_file_train = os.path.join(path_saludable, "Aventa_sensors.json")
    dataset_name = "Aventa"

    # Cargar el dataset con falla
    path_con_falla = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/"
    file_path_test = os.path.join(path_con_falla, "Aventa_Taggenberg_16_02_2022.hdf5")
    json_file_test = os.path.join(path_con_falla, "Aventa_sensors.json")
    dataset_name = "Aventa"

    reader_train = HDF5Reader(file_path_train, json_file_train)
    # Obtener los índices de corte basados en el tamaño del timestamp de entrenamiento
    train_timestamps = reader_train.print_timestamps(dataset_name)
    ### Solo prueba
    train_timestamps = train_timestamps[-4:]
    ###
    num_train_samples = len(train_timestamps)
    train_end = int(num_train_samples * 0.6)
    calibration_end = int(num_train_samples * 0.8)

    # Dividir el dataset de entrenamiento en función de los índices de corte
    print("Cargando el dataset de entrenamiento")
    df_train = reader_train.load_all_signals_for_timestamps(dataset_name, train_timestamps[:train_end])
    print("Cargando el dataset de calibración")
    df_calibration = reader_train.load_all_signals_for_timestamps(dataset_name, train_timestamps[train_end:calibration_end])
    print("Cargando el dataset de validación")
    df_valid = reader_train.load_all_signals_for_timestamps(dataset_name, train_timestamps[calibration_end:])

    df_train = df_train.drop(columns=['Time', 'Timestamp'])  # Eliminar columnas no necesarias
    df_calibration = df_calibration.drop(columns=['Time', 'Timestamp'])  # Eliminar columnas no necesarias
    df_valid = df_valid.drop(columns=['Time', 'Timestamp'])  # Eliminar columnas no necesarias

    reader_test = HDF5Reader(file_path_test, json_file_test)
    test_timestamps = reader_test.print_timestamps(dataset_name)
    ### Solo para probar minimizar el dataset de prueba
    test_timestamps = test_timestamps[-62:-32] #-32
    ###
    #print(test_timestamps)
    print("Cargando el dataset de prueba")
    df_test = reader_test.load_all_signals_for_timestamps(dataset_name, test_timestamps)
    df_test = df_test.drop(columns=['Time'])  # Eliminamos solo 'Time', conservamos 'Timestamp'
    df_test1 = df_test.copy()
    df_test = df_test.drop(columns=['Timestamp'])
    
    print(df_test.shape, df_train.shape, df_calibration.shape, df_valid.shape)
    
    ######### Solo a modo de prueba
    #df_train = df_train.iloc[-1000:, :]
    #df_calibration = df_calibration.iloc[-1000:, :]
    #df_valid = df_valid.iloc[-1000:, :]
    #df_test = df_test.iloc[-1000:, :]
    #exit()

    # Normalizar los datasets
    print("Normalizando los datasets")
    scaler = MinMaxScaler()
    df_train_normalized = scaler.fit_transform(df_train)
    df_val_normalized = scaler.transform(df_valid)
    df_calibration_normalized = scaler.transform(df_calibration)
    df_test_normalized = scaler.transform(df_test)
    

    if not args.only_testing:
        # Entrenar el autoencoder con el conjunto de entrenamiento y validación
        print("Entrenando el autoencoder con early stopping")
        input_size = df_train.shape[1]
        model = AutoencoderMLP(input_size).to(device)
        trained_model, loss_history, val_loss_history = train_autoencoder(
            model,
            df_train_normalized,
            val_data=df_val_normalized,
            epochs=100,
            learning_rate=0.001,
            batch_size=32,
            patience=5,
            device=device
        )

        # Guardar el modelo entrenado
        torch.save(trained_model.state_dict(), 'output/best_model.pth')

        # Guardar gráficos de pérdida
        plt.figure()
        plt.plot(loss_history, label='Pérdida de entrenamiento')
        plt.plot(val_loss_history, label='Pérdida de validación')
        plt.title('Función de pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.savefig('output/loss_plot.png')
        plt.close()
    else:
        # Verificar que el modelo guardado exista
        if not os.path.exists('output/best_model.pth'):
            print("El modelo 'best_model.pth' no existe en la carpeta 'output'. Por favor, entrena el modelo primero.")
            return

    # Cargar el mejor modelo guardado
    print("Cargando el modelo")
    input_size = df_train.shape[1]
    best_model = AutoencoderMLP(input_size).to(device)
    best_model.load_state_dict(torch.load('output/best_model.pth'))
    best_model.eval()

    # Detección de anomalías en el conjunto de prueba
    print("Realizando detección de anomalías en el conjunto de prueba")
    reconstruction_error, q_hat = conformal_anomaly_detection(
        best_model,
        df_test_normalized,
        calibration_data=df_calibration_normalized,
        significance_level=0.05,
        device=device
    )

    # Crear el gráfico de detección de anomalías
    df_test['Timestamp'] = df_test1['Timestamp']

    # Obtener el número de muestras por timestamp
    num_muestras_por_timestamp = len(reconstruction_error) // len(test_timestamps)

    # Calcular las posiciones en el eje x donde inicia cada timestamp
    xticks_positions = [i * num_muestras_por_timestamp for i in range(len(test_timestamps))]

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_error, label='Error de reconstrucción')
    plt.axhline(y=q_hat, color='r', linestyle='--', label='Umbral de anomalía')
    plt.title('Detección de anomalías en conjunto de prueba')
    plt.xlabel('Muestras')
    plt.ylabel('Error de reconstrucción')
    plt.legend()

    # Establecer los ticks en las posiciones calculadas y utilizar 'test_timestamps' como etiquetas
    plt.xticks(ticks=xticks_positions, labels=test_timestamps, rotation='vertical')

    plt.tight_layout()
    plt.savefig('output/anomaly_detection.png')
    plt.close()

    print("Proceso completado. Los resultados se han guardado en la carpeta 'output'.")

if __name__ == "__main__":
    main()
