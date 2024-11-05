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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse  # Importación añadida
from conformal_anomaly_detector import ConformalAnomalyDetector, ManualADPredictor


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

    # Cargar el dataset saludable
    path_saludable = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/"
    file_path_train = os.path.join(path_saludable, "Aventa_Taggenberg_15_02_2022.hdf5")
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
    train_timestamps = train_timestamps[-10:]
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
    
    # Aplicar PCA con 10 componentes
    print("Aplicando PCA con 6 componentes")
    pca = PCA(n_components=6)
    df_train_pca = pca.fit_transform(df_train_normalized)
    df_val_pca = pca.transform(df_val_normalized)
    df_calibration_pca = pca.transform(df_calibration_normalized)
    df_test_pca = pca.transform(df_test_normalized)

    # Actualizar input_size para el autoencoder
    input_size = df_train_pca.shape[1]

    if not args.only_testing:
        # Entrenar el autoencoder con el conjunto de entrenamiento y validación
        print("Entrenando el autoencoder con early stopping")
        model = AutoencoderMLP(input_size).to(device)
        trained_model, loss_history, val_loss_history = train_autoencoder(
            model,
            df_train_pca,
            val_data=df_val_pca,
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
    best_model = AutoencoderMLP(input_size).to(device)
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
    q_hat_05 = manual_cad_05.fit(train_data=df_train_pca, calibration_data=df_calibration_pca, alpha=0.05)
    q_hat_10 = manual_cad_10.fit(train_data=df_train_pca, calibration_data=df_calibration_pca, alpha=0.1)

    # Predict anomalies using the manual CAD
    manual_cad_results_05 = manual_cad_05.predict(df_test_pca)
    manual_cad_results_10 = manual_cad_10.predict(df_test_pca)
    manual_cad_anomalies_05 = df_test_pca[manual_cad_results_05]
    manual_cad_not_anomalies_05 = df_test_pca[~manual_cad_results_05]
    manual_cad_anomalies_10 = df_test_pca[manual_cad_results_10]
    manual_cad_not_anomalies_10 = df_test_pca[~manual_cad_results_10]
    
    print("########################################################")
    print(manual_cad_results_05.shape, manual_cad_anomalies_05.shape, manual_cad_not_anomalies_05.shape)
    print(manual_cad_results_10.shape, manual_cad_anomalies_10.shape, manual_cad_not_anomalies_10.shape)
    print("########################################################")
    print(f"Umbral de anomalía (alpha=0.05): {q_hat_05}")
    print(f"Umbral de anomalía (alpha=0.1): {q_hat_10}")
    reconstruction_error = manual_cad_05.predictor.predict(df_test_pca)

    # Guardar reconstruction_error en CSV
    df_reconstruction = pd.DataFrame({
        'Timestamp': df_test1['Timestamp'],
        'Error_Reconstrucción': reconstruction_error
    })
    # Añadir columna 'Anomalia_05' donde 1 si supera el umbral de 0.05, 0 en caso contrario
    df_reconstruction['Anomalia_05'] = (df_reconstruction['Error_Reconstrucción'] > q_hat_05).astype(int)
    # Añadir columna 'Anomalia_10' donde 1 si supera el umbral de 0.1, 0 en caso contrario
    df_reconstruction['Anomalia_10'] = (df_reconstruction['Error_Reconstrucción'] > q_hat_10).astype(int)
    ruta_reconstruction_csv = os.path.join('output', 'reconstruction_error.csv')
    df_reconstruction.to_csv(ruta_reconstruction_csv, index=False)
    print(f"Errores de reconstrucción guardados en: {ruta_reconstruction_csv}")

    # Crear el gráfico de detección de anomalías
    df_test['Timestamp'] = df_test1['Timestamp']

    # Obtener el número de muestras por timestamp
    num_muestras_por_timestamp = len(reconstruction_error) // len(test_timestamps)

    # Calcular las posiciones en el eje x donde inicia cada timestamp
    xticks_positions = [i * num_muestras_por_timestamp for i in range(len(test_timestamps))]

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_error, label='Error de reconstrucción')
    plt.axhline(y=q_hat_05, color='r', linestyle='--', label='Umbral de anomalía (alpha=0.05)')
    #plt.axhline(y=q_hat_10, color='g', linestyle='--', label='Umbral de anomalía (alpha=0.1)')
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
