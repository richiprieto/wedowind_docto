import torch
import numpy as np
import random
import logging
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
from autoencoder_mlp import AutoencoderMLP
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
from conformal_anomaly_detector import ConformalAnomalyDetector, ManualADPredictor
from datetime import datetime
from autoencoder_kan import AutoencoderKAN
import h5py
import re
import gc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log'),
        logging.StreamHandler()
    ]
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_merge_datasets(reader, dataset_name, file_paths):
    """Carga y fusiona múltiples archivos HDF5 en orden cronológico"""
    full_df = pd.DataFrame()
    for file_path in file_paths:
        try:
            reader.file_path = file_path  # Actualizar path en el reader
            timestamps = reader.print_timestamps(dataset_name, print_timestamps=False)
            df = reader.load_all_signals_for_timestamps(dataset_name, timestamps)
            full_df = pd.concat([full_df, df], ignore_index=True)
            logging.info(f"Archivo {os.path.basename(file_path)} cargado exitosamente")
            del df  # Liberar memoria
            gc.collect()  # Recolectar basura
        except Exception as e:
            logging.error(f"Error cargando {file_path}: {str(e)}")
            raise
    return full_df.sort_values('Timestamp')

def create_windowed_timestamps(timestamps, window_size, step_size):
    windowed_timestamps = []
    for start in range(0, len(timestamps) - window_size + 1, step_size):
        central_index = start + window_size // 2
        windowed_timestamps.append(timestamps.iloc[central_index])
    return windowed_timestamps

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento y detección de anomalías con Autoencoder")
    parser.add_argument('--only_testing', action='store_true', help='Modo solo prueba')
    parser.add_argument('--test_dataset', action='store_true', help='Usar dataset original de prueba')
    args = parser.parse_args()

    set_seed()
    # Configurar dispositivo: GPU solo para entrenamiento
    device = "cuda" if torch.cuda.is_available() and not args.only_testing else "cpu"
    logging.info(f"Iniciando ejecución en dispositivo: {device}")
    os.makedirs('output', exist_ok=True)

    # Configuración de paths según el flag
    if args.test_dataset:
        logging.info("Usando dataset de prueba original")
        base_path = "../../primer_modelo"
        train_files = [os.path.join(base_path, "Aventa_Taggenberg_16_02_2022.hdf5")]
        test_file = os.path.join(base_path, "Aventa_Taggenberg_16_02_2022.hdf5")
        json_file = os.path.join(base_path, "Aventa_sensors.json")
    else:
        logging.info("Usando nuevo conjunto de datasets")
        base_path = "../../primer_modelo"
        train_files = [
            os.path.join(base_path, "Aventa_Taggenberg_06_02_2022.hdf5"),
            os.path.join(base_path, "Aventa_Taggenberg_11_02_2022.hdf5"),
            os.path.join(base_path, "Aventa_Taggenberg_14_02_2022.hdf5"),
            os.path.join(base_path, "Aventa_Taggenberg_15_02_2022.hdf5")
        ]
        test_file = os.path.join(base_path, "Aventa_Taggenberg_16_02_2022.hdf5")
        json_file = os.path.join(base_path, "Aventa_sensors.json")

    # Cargar datos de entrenamiento/validación
    full_train_df = None
    try:
        reader = HDF5Reader(train_files[0], json_file)
        dataset_name = "Aventa"

        if args.test_dataset:
            # Carga original
            train_timestamps = reader.print_timestamps(dataset_name, print_timestamps=False)
            subset_size = 100
            train_subset = train_timestamps[:subset_size][:10]
            
            # Dividir subsets
            train_end = int(0.8 * len(train_subset))
            calibration_end = int(0.9 * len(train_subset))
            validation_end = len(train_subset)
            
            # Cargar datos
            df_train = reader.load_all_signals_for_timestamps(dataset_name, train_subset[:train_end]).drop(columns=['Time', 'Timestamp'], errors='ignore')
            df_calibration = reader.load_all_signals_for_timestamps(dataset_name, train_subset[train_end:calibration_end]).drop(columns=['Time', 'Timestamp'], errors='ignore')
            df_valid = reader.load_all_signals_for_timestamps(dataset_name, train_subset[calibration_end:validation_end]).drop(columns=['Time', 'Timestamp'], errors='ignore')

        else:
            # Carga y fusión de nuevos datasets
            logging.info("Cargando y fusionando datasets de entrenamiento")
            full_train_df = load_and_merge_datasets(reader, dataset_name, train_files)
            total_samples = len(full_train_df)
            
            # Dividir en train/cal/val
            train_end = int(0.8 * total_samples)
            calibration_end = int(0.9 * total_samples)
            
            df_train = full_train_df.iloc[:train_end].drop(columns=['Time', 'Timestamp'], errors='ignore')
            df_calibration = full_train_df.iloc[train_end:calibration_end].drop(columns=['Time', 'Timestamp'], errors='ignore')
            df_valid = full_train_df.iloc[calibration_end:].drop(columns=['Time', 'Timestamp'], errors='ignore')
            
            del full_train_df  # Liberar memoria
            gc.collect()  # Recolectar basura

    except Exception as e:
        logging.error(f"Error inicializando datasets: {str(e)}")
        return

    # Resto del procesamiento (igual para ambos casos)
    logging.info("Normalizando datos")
    scaler = MinMaxScaler()
    if args.only_testing:
        df_train_normalized = scaler.fit_transform(df_train)
        df_calibration_normalized = scaler.transform(df_calibration)
        del df_train, df_valid, df_train_normalized  # Liberar memoria
        gc.collect()  # Recolectar basura
    else:
        df_train_normalized = scaler.fit_transform(df_train)
        df_val_normalized = scaler.transform(df_valid)
        df_calibration_normalized = scaler.transform(df_calibration)
        del df_train, df_valid  # Liberar memoria
        gc.collect()  # Recolectar basura
    
    window_size = 400
    step_size = 400

    def create_windows(data, window_size, step_size):
        windows = []
        for start in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[start:start+window_size].flatten())
        return np.array(windows)

    logging.info("Creando ventanas de datos")
    df_calibration_windows = create_windows(df_calibration_normalized, window_size, step_size)
    input_size = df_calibration_windows.shape[1]
    
    del df_calibration_normalized  # Liberar memoria
    gc.collect()  # Recolectar basura

    if not args.only_testing:
        df_train_windows = create_windows(df_train_normalized, window_size, step_size)
        df_val_windows = create_windows(df_val_normalized, window_size, step_size)
        del df_train_normalized, df_val_normalized  # Liberar memoria
        gc.collect()  # Recolectar basura
        
        logging.info("Entrenando modelo")
        #model = AutoencoderKAN(input_size).to(device)
        model = AutoencoderMLP(input_size).to(device)
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

        # Mover modelo a CPU antes de guardar
        trained_model.cpu()
        mejor_epoca = np.argmin(val_loss_history) + 1
        for archivo in os.listdir('output'):
            if archivo.endswith('.pth'):
                os.remove(os.path.join('output', archivo))
        nombre_modelo = f"best_model_ep{mejor_epoca}_val{val_loss_history[mejor_epoca-1]:.6f}.pth"
        torch.save(trained_model.state_dict(), os.path.join('output', nombre_modelo))

        plt.figure()
        plt.plot(loss_history, label='train loss')
        plt.plot(val_loss_history, label='validation loss')
        plt.legend()
        plt.savefig('output/loss_plot.png')
        plt.close()

        pd.DataFrame({
            'Época': range(1, len(loss_history)+1),
            'Pérdida_Entrenamiento': loss_history,
            'Pérdida_Validación': val_loss_history
        }).to_csv('output/loss_history.csv', index=False)
        
        del trained_model, loss_history, val_loss_history, df_train_windows, df_val_windows  # Liberar memoria
        gc.collect()  # Recolectar basura

    modelos_existentes = [f for f in os.listdir('output') if f.endswith('.pth')]
    if not modelos_existentes:
        print("No hay modelos guardados")
        return

    logging.info("Cargando mejor modelo")
    # Forzar carga en CPU aunque se haya entrenado en GPU
    #best_model = AutoencoderKAN(input_size).to('cpu')
    best_model = AutoencoderMLP(input_size).to('cpu')
    best_model.load_state_dict(
        torch.load(
            os.path.join('output', modelos_existentes[0]),
            map_location=torch.device('cpu')  # Asegurar carga en CPU
        )
    )
    best_model.eval()


    logging.info("Entrenando detector de anomalías")
    manual_predictor = ManualADPredictor(best_model)
    manual_cad_05 = ConformalAnomalyDetector(manual_predictor)
    manual_cad_10 = ConformalAnomalyDetector(manual_predictor)
    q_hat_05 = manual_cad_05.fit(df_calibration_windows, df_calibration_windows, 0.05)
    q_hat_10 = manual_cad_10.fit(df_calibration_windows, df_calibration_windows, 0.1)
    
    del df_calibration_windows  # Liberar memoria
    gc.collect()  # Recolectar basura

    # Antes del bucle de procesamiento
    batch_size = 30
    test_reader = HDF5Reader(test_file, json_file)
    test_timestamps = test_reader.print_timestamps(dataset_name, print_timestamps=False)
    total_timestamps = len(test_timestamps)
    test_subset = test_timestamps
    test_batches = [(i, min(i + batch_size, total_timestamps)) 
                    for i in range(0, total_timestamps, batch_size)]
    
    # Procesamiento por batches de prueba
    for batch_num, (batch_start, batch_end) in enumerate(test_batches, 1):
        logging.info(f"Procesando batch {batch_num}: timestamps {batch_start}-{batch_end-1}")
        
        batch_subset = test_subset[batch_start:batch_end]
        df_test = test_reader.load_all_signals_for_timestamps(dataset_name, batch_subset)
        if df_test.empty:
            logging.warning(f"Batch {batch_num} vacío")
            continue
            
        df_test1 = df_test.copy()
        df_test_clean = df_test.drop(columns=['Time', 'Timestamp'], errors='ignore')
        df_test_normalized = scaler.transform(df_test_clean)
        
        del df_test, df_test_clean  # Liberar memoria
        gc.collect()  # Recolectar basura
        
        df_test_windows = create_windows(df_test_normalized, window_size, step_size)
        windowed_timestamps = pd.Series(create_windowed_timestamps(df_test1['Timestamp'], window_size, step_size))
        
        del df_test_normalized, df_test1  # Liberar memoria
        gc.collect()  # Recolectar basura
        
        if df_test_windows.size == 0:
            logging.warning(f"Batch {batch_num} no tiene ventanas válidas")
            continue

        reconstruction_error = manual_cad_05.predictor.predict(df_test_windows)
        anomalies_05 = reconstruction_error > q_hat_05
        anomalies_10 = reconstruction_error > q_hat_10
        
        del df_test_windows  # Liberar memoria
        gc.collect()  # Recolectar basura

        # Guardar CSV por batch
        df_batch = pd.DataFrame({
            'Timestamp': windowed_timestamps,
            'Error_Reconstrucción': reconstruction_error,
            'Anomalia_05': anomalies_05.astype(int),
            'Anomalia_10': anomalies_10.astype(int)
        })
        batch_csv_path = os.path.join('output', f'reconstruction_batch_{batch_num}.csv')
        df_batch.to_csv(batch_csv_path, index=False)
        logging.info(f"CSV guardado: {batch_csv_path}")
        
        del df_batch  # Liberar memoria
        gc.collect()  # Recolectar basura

        # Generar gráfico por batch
        plt.figure(figsize=(12, 6))
        plt.plot(reconstruction_error, label='Error de reconstrucción')
        plt.axhline(q_hat_05, color='r', linestyle='--', label='Umbral (α=0.05)')
        plt.axhline(q_hat_10, color='g', linestyle='--', label='Umbral (α=0.1)')
        
        unique_indices = []
        unique_timestamps = []
        prev_ts = None
        for idx, ts in enumerate(windowed_timestamps):
            if ts != prev_ts:
                unique_indices.append(idx)
                unique_timestamps.append(ts)
                prev_ts = ts
                
        plt.xticks(unique_indices, unique_timestamps, rotation=45, ha='right')
        plt.title(f'Detección de anomalías - Batch {batch_num}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('output', f'anomaly_batch_{batch_num}.png'))
        plt.close()
        
        del reconstruction_error, anomalies_05, anomalies_10, windowed_timestamps  # Liberar memoria
        gc.collect()  # Recolectar basura

    logging.info("Proceso completado. Resultados guardados en 'output'.")

if __name__ == "__main__":
    main()