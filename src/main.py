import torch
import numpy as np
import random
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
#from conformal_detection import conformal_anomaly_detection
from autoencoder_mlp import AutoencoderMLP
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse  # Importación añadida
from conformal_anomaly_detector import ConformalAnomalyDetector, ManualADPredictor
from datetime import datetime
from autoencoder_kan import AutoencoderKAN
import h5py
import re

#######################
#from deel.puncc.api.prediction import BasePredictor
#from deel.puncc.anomaly_detection import SplitCAD
#######################

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

    #base_path = "../"
    
    # Obtener el directorio actual
    current_dir = os.getcwd()
    print(f"Directorio actual: {current_dir}")

    # Cargar el dataset saludable
    path_saludable = "../../primer_modelo"
    
    # Listar archivos en el directorio
    if os.path.exists(path_saludable):
        print("\nArchivos en el directorio:")
        for file in os.listdir(path_saludable):
            print(f"- {file}")
    else:
        print(f"El directorio {path_saludable} no existe")

    file_path_train = os.path.join(path_saludable, "Aventa_Taggenberg_16_02_2022.hdf5")
    json_file_train = os.path.join(path_saludable, "Aventa_sensors.json")
    dataset_name = "Aventa"

    if not os.path.exists(file_path_train):
        print(f"Error: File not found at {file_path_train}")
        return
    if not os.path.exists(json_file_train):
        print(f"Error: JSON file not found at {json_file_train}")
        return

    reader_train = HDF5Reader(file_path_train, json_file_train)
    
    # Obtener todos los timestamps del dataset
    train_timestamps = reader_train.print_timestamps(dataset_name, print_timestamps=False)
    print(f"Total de timestamps disponibles: {len(train_timestamps)}")

    # Verificar que hay suficientes timestamps
    if len(train_timestamps) < 100:
        print(f"Error: Se requieren al menos 100 timestamps para la división, pero solo se encontraron {len(train_timestamps)}.")
        return

    # Dividir el dataset en dos partes:
    # - Primeros 100 timestamps para entrenamiento, calibración y validación
    # - Resto de los timestamps para prueba
    subset_size = 100
    train_subset = train_timestamps[:subset_size]
    # Solo para probar minimizar el dataset de entrenamiento, cuestiones de tiempo
    train_subset = train_subset[:10]
   

    #test_subset = train_timestamps[subset_size:subset_size+10]
    #test_subset = train_timestamps[len(train_subset):len(train_subset)+10]
    test_subset = train_timestamps[58:64]
    # Dividir los primeros 100 timestamps en entrenamiento, calibración y validación
    train_end = int(0.8 * len(train_subset))  # 80 timestamps para entrenamiento
    print(train_end)
    calibration_end = int(0.9 * len(train_subset))  # 10 timestamps para calibración
    print(calibration_end)
    validation_end = len(train_subset)  # 10 timestamps para validación

    print("Dividiendo el dataset en conjuntos de entrenamiento, calibración, validación y prueba")

    # Cargar el dataset de entrenamiento
    print("Cargando el dataset de entrenamiento")
    df_train = reader_train.load_all_signals_for_timestamps(dataset_name, train_subset[:train_end])
    if df_train.empty:
        print("Error: No se pudieron cargar datos para el conjunto de entrenamiento.")
        return

    # Cargar el dataset de calibración
    print("Cargando el dataset de calibración")
    df_calibration = reader_train.load_all_signals_for_timestamps(dataset_name, train_subset[train_end:calibration_end])
    if df_calibration.empty:
        print("Error: No se pudieron cargar datos para el conjunto de calibración.")
        return

    # Cargar el dataset de validación
    print("Cargando el dataset de validación")
    df_valid = reader_train.load_all_signals_for_timestamps(dataset_name, train_subset[calibration_end:validation_end])
    if df_valid.empty:
        print("Error: No se pudieron cargar datos para el conjunto de validación.")
        return

    # Cargar el dataset de prueba
    print("Cargando el dataset de prueba")
    df_test = reader_train.load_all_signals_for_timestamps(dataset_name, test_subset)
    if df_test.empty:
        print("Error: No se pudieron cargar datos para el conjunto de prueba.")
        return

    # Eliminar columnas no necesarias
    df_train = df_train.drop(columns=['Time', 'Timestamp'], errors='ignore')
    df_calibration = df_calibration.drop(columns=['Time', 'Timestamp'], errors='ignore')
    df_valid = df_valid.drop(columns=['Time', 'Timestamp'], errors='ignore')
    df_test1 = df_test.copy()
    df_test = df_test.drop(columns=['Time', 'Timestamp'], errors='ignore')

    print(f"Dimensiones de los conjuntos de datos:")
    print(f"Entrenamiento: {df_train.shape}")
    print(f"Calibración: {df_calibration.shape}")
    print(f"Validación: {df_valid.shape}")
    print(f"Prueba: {df_test.shape}")

    # Normalizar los datasets
    print("Normalizando los datasets")
    scaler = MinMaxScaler()
    df_train_normalized = scaler.fit_transform(df_train)
    df_val_normalized = scaler.transform(df_valid)
    df_calibration_normalized = scaler.transform(df_calibration)
    df_test_normalized = scaler.transform(df_test)

    # Parámetros de ventana
    window_size = 400
    step_size = 400

    # Función para crear ventanas
    def create_windows(data, window_size, step_size):
        windows = []
        for start in range(0, len(data) - window_size + 1, step_size):
            end = start + window_size
            window = data[start:end].flatten()  # Aplanar cada ventana para usar con capas lineales
            windows.append(window)
        return np.array(windows)
    
    def create_windowed_timestamps(timestamps, window_size, step_size):
        windowed_timestamps = []
        total_windows = (len(timestamps) - window_size) // step_size + 1
        for i in range(total_windows):
            start = i * step_size
            end = start + window_size
            # Usar el timestamp central de la ventana
            central_index = start + window_size // 2
            if central_index < len(timestamps):
                windowed_timestamps.append(timestamps.iloc[central_index])
            else:
                windowed_timestamps.append(timestamps.iloc[-1])
        return windowed_timestamps
    
         # Crear timestamps ajustados para las ventanas de prueba
    print("Ajustando timestamps para las ventanas de prueba")
    windowed_timestamps = pd.Series(create_windowed_timestamps(df_test1['Timestamp'], window_size, step_size))
    print(f"Número de timestamps ajustados: {len(windowed_timestamps)}")

    # Crear ventanas
    df_train_windows = create_windows(df_train_normalized, window_size, step_size)
    df_val_windows = create_windows(df_val_normalized, window_size, step_size)
    df_calibration_windows = create_windows(df_calibration_normalized, window_size, step_size)
    df_test_windows = create_windows(df_test_normalized, window_size, step_size)
    print("Forma de las ventanas:")
    print(f"Entrenamiento: {df_train_windows.shape}")
    print(f"Validación: {df_val_windows.shape}")
    print(f"Calibración: {df_calibration_windows.shape}")
    print(f"Prueba: {df_test_windows.shape}")

    # Dimensión de entrada
    input_size = df_train_windows.shape[1]

    if not args.only_testing:

        # Entrenar el autoencoder con el conjunto de entrenamiento y validación
        print("Entrenando el autoencoder con early stopping")
        model = AutoencoderKAN(input_size).to(device)
        #model = AutoencoderMLP(input_size).to(device)
        trained_model, loss_history, val_loss_history = train_autoencoder(
            model,
            df_train_windows,
            val_data=df_val_windows,
            epochs=100,
            learning_rate=0.001,
            batch_size=32,
            patience=3,
            device=device
        )

        # Determinar la mejor época basada en el menor valor de pérdida de validación
        mejor_epoca = np.argmin(val_loss_history) + 1  # +1 si las épocas inician en 1
        mejor_val_loss = val_loss_history[mejor_epoca - 1]

        # Eliminar cualquier archivo .pth existente en la carpeta 'output'
        for archivo in os.listdir('output'):
            if archivo.endswith('.pth'):
                ruta_archivo = os.path.join('output', archivo)
                os.remove(ruta_archivo)
                print(f"Eliminado archivo existente: {ruta_archivo}")

        # Guardar el modelo entrenado con el número de época y el valor de pérdida de validación
        nombre_modelo = f"best_model_ep{mejor_epoca}_val{mejor_val_loss:.6f}.pth"
        ruta_modelo = os.path.join('output', nombre_modelo)
        torch.save(trained_model.state_dict(), ruta_modelo)
        print(f"Modelo guardado como: {ruta_modelo}")

        # Guardar gráficos de pérdida
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='train loss')
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='validation loss')
        plt.title('Función de pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.savefig('output/loss_plot.png')
        plt.close()

        # Guardar loss_history y val_loss_history en CSV
        df_loss = pd.DataFrame({
            'Época': list(range(1, len(loss_history) + 1)),
            'Pérdida_Entrenamiento': loss_history,
            'Pérdida_Validación': val_loss_history
        })
        ruta_loss_csv = os.path.join('output', 'loss_history.csv')
        df_loss.to_csv(ruta_loss_csv, index=False)
        print(f"Historial de pérdidas guardado en: {ruta_loss_csv}")
    else:
        # Verificar que el modelo guardado exista
        modelos_existentes = [archivo for archivo in os.listdir('output') if archivo.endswith('.pth')]
        if not modelos_existentes:
            print("No existen modelos guardados en la carpeta 'output'. Por favor, entrena el modelo primero.")
            return

    # Cargar el mejor modelo guardado
    modelos_existentes = [archivo for archivo in os.listdir('output') if archivo.endswith('.pth')]
    print("Cargando el modelo")
    #best_model = AutoencoderMLP(input_size).to(device)
    best_model = AutoencoderKAN(input_size).to(device)
    best_model.load_state_dict(torch.load('output/'+modelos_existentes[0]))
    best_model.eval()

    # Detección de anomalías en el conjunto de prueba
    print("Realizando detección de anomalías en el conjunto de prueba")
    #reconstruction_error, q_hat = conformal_anomaly_detection(
    #    best_model,
    #    df_test_pca,
    #    calibration_data=df_calibration_pca,
    #    significance_level=0.05,
    #    device=device
    #)

    # ----------------------------
    # Predictor for deel.puncc (SplitCAD)
    # ----------------------------
    #class ADPredictor(BasePredictor):
    #    def predict(self, X):
    #        return -self.model.score_samples(X)

    # Wrap the Isolation Forest model in the predictor
    #if_predictor = ADPredictor(best_model)

    # Instantiate SplitCAD on top of the predictor
    #split_cad = SplitCAD(if_predictor, train=True, random_state=0)

    # Fit SplitCAD on the dataset with a fit ratio of 0.7
    #split_cad.fit(z=df_calibration_pca, fit_ratio=0.9)

    # Predict anomalies using SplitCAD with alpha = 0.05
    #alpha = 0.05
    #split_cad_results = split_cad.predict(df_test_pca, alpha=alpha)
    #split_cad_anomalies = df_test_pca[split_cad_results]
    #split_cad_not_anomalies = df_test_pca[np.invert(split_cad_results)]

    #print("########################################################")
    #print(split_cad_results.shape, split_cad_anomalies.shape, split_cad_not_anomalies.shape)
    #print("########################################################")
    #print(split_cad_anomalies)
    #print("########################################################")

    # Initialize the manual predictor and Conformal Anomaly Detector
    manual_predictor = ManualADPredictor(best_model)
    manual_cad_05 = ConformalAnomalyDetector(manual_predictor)
    manual_cad_10 = ConformalAnomalyDetector(manual_predictor)

    # Fit the manual CAD on la dataset
    q_hat_05 = manual_cad_05.fit(train_data=df_train_windows, calibration_data=df_calibration_windows, alpha=0.05)
    q_hat_10 = manual_cad_10.fit(train_data=df_train_windows, calibration_data=df_calibration_windows, alpha=0.1)

    df_test_windows = df_test_windows #pd.concat([df_val_windows, df_test_windows])
    # Predict anomalies using the manual CAD
    manual_cad_results_05 = manual_cad_05.predict(df_test_windows)
    manual_cad_results_10 = manual_cad_10.predict(df_test_windows)
    manual_cad_anomalies_05 = df_test_windows[manual_cad_results_05]
    manual_cad_not_anomalies_05 = df_test_windows[~manual_cad_results_05]
    manual_cad_anomalies_10 = df_test_windows[manual_cad_results_10]
    manual_cad_not_anomalies_10 = df_test_windows[~manual_cad_results_10]
    
    print("########################################################")
    print(manual_cad_results_05.shape, manual_cad_anomalies_05.shape, manual_cad_not_anomalies_05.shape)
    print(manual_cad_results_10.shape, manual_cad_anomalies_10.shape, manual_cad_not_anomalies_10.shape)
    print("########################################################")
    print(f"Umbral de anomalía (alpha=0.05): {q_hat_05}")
    print(f"Umbral de anomalía (alpha=0.1): {q_hat_10}")
    reconstruction_error = manual_cad_05.predictor.predict(df_test_windows)

    # Guardar reconstruction_error en CSV
    df_reconstruction = pd.DataFrame({
        'Timestamp': windowed_timestamps,
        'Error_Reconstrucción': reconstruction_error
    })
    # Añadir columna 'Anomalia_05' donde 1 si supera el umbral de 0.05, 0 en caso contrario
    df_reconstruction['Anomalia_05'] = (df_reconstruction['Error_Reconstrucción'] > q_hat_05).astype(int)
    # Añadir columna 'Anomalia_10' donde 1 si supera el umbral de 0.1, 0 en caso contrario
    df_reconstruction['Anomalia_10'] = (df_reconstruction['Error_Reconstrucción'] > q_hat_10).astype(int)
    ruta_reconstruction_csv = os.path.join('output', 'reconstruction_error.csv')
    df_reconstruction.to_csv(ruta_reconstruction_csv, index=False)
    print(f"Errores de reconstrucción guardados en: {ruta_reconstruction_csv}")

        # Usar los timestamps ajustados
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_error, label='Error de reconstrucción')
    plt.axhline(y=q_hat_05, color='r', linestyle='--', label='Umbral de anomalía (alpha=0.05)')
    plt.axhline(y=q_hat_10, color='g', linestyle='--', label='Umbral de anomalía (alpha=0.1)')
    plt.title('Detección de anomalías en conjunto de prueba')
    plt.xlabel('Muestras')
    plt.ylabel('Error de reconstrucción')
    plt.legend()

    # Identificar los índices de las primeras ocurrencias de cada timestamp único
    unique_timestamps = []
    unique_indices = []
    previous_ts = None
    for idx, ts in enumerate(windowed_timestamps):
        if ts != previous_ts:
            unique_timestamps.append(ts)
            unique_indices.append(idx)
            previous_ts = ts

    # Establecer los ticks en las posiciones identificadas y asignar sus etiquetas
    plt.xticks(ticks=unique_indices, labels=unique_timestamps, rotation='vertical')

    plt.tight_layout()
    plt.savefig('output/anomaly_detection.png')
    plt.close()

    print("Proceso completado. Los resultados se han guardado en la carpeta 'output'.")

if __name__ == "__main__":
    main()
