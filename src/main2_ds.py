import torch
import numpy as np
import random
import logging
from read_hdf5 import HDF5Reader
from train_autoencoder import train_autoencoder
from autoencoder_kan import AutoencoderKAN
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
from conformal_anomaly_detector import ConformalAnomalyDetector, ManualADPredictor
import h5py
import re
import gc
from tqdm import tqdm
from datetime import datetime
import joblib

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Configura la semilla para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Semilla configurada a {seed}")

def create_windows(data, window_size, step_size):
    """Crea ventanas de datos con tamaño y paso especificados"""
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start+window_size].flatten())
    return np.array(windows, dtype=np.float32) if windows else np.empty((0, window_size * data.shape[1]))

def create_windowed_timestamps(timestamps, window_size, step_size):
    """Crea timestamps correspondientes a las ventanas"""
    return [
        timestamps.iloc[i * step_size + window_size // 2] 
        if (i * step_size + window_size // 2) < len(timestamps) 
        else timestamps.iloc[-1] 
        for i in range((len(timestamps) - window_size) // step_size + 1)
    ]

class GPUMemoryManager:
    """Clase para gestión de memoria en GPU"""
    
    @staticmethod
    def auto_batch_size(device, model, sample_size, safety_margin=0.2):
        """Calcula el batch size máximo según la memoria disponible"""
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                total_mem = torch.cuda.get_device_properties(device).total_memory
                reserved_mem = torch.cuda.memory_reserved(device)
                free_mem = total_mem - reserved_mem
                
                dummy_input = torch.randn(2, sample_size, device=device)
                model(dummy_input)
                mem_per_sample = (torch.cuda.memory_allocated(device) - reserved_mem) / 2
                
                max_batch = int((free_mem * safety_margin) / mem_per_sample)
                return max(1, max_batch)
                
            except Exception as e:
                logger.warning(f"Error en estimación de memoria: {str(e)}")
                return 8
        return 32

def main():
    """Función principal de ejecución"""
    start_time = datetime.now()
    logger.info("Iniciando ejecución del sistema de detección de anomalías")
    
    parser = argparse.ArgumentParser(description="Sistema de detección de anomalías con Autoencoder KAN")
    parser.add_argument('--only_testing', action='store_true', help='Ejecutar solo fase de inferencia')
    args = parser.parse_args()

    set_seed()
    dataset_name = "Aventa"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo seleccionado: {device}")
    
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)

    path_saludable = "../../primer_modelo"
    json_file = os.path.join(path_saludable, "Aventa_sensors.json")
    test_file = "Aventa_Taggenberg_16_02_2022.hdf5"

    window_size, step_size = 400, 400
    scaler = MinMaxScaler()
    all_windows = []
    train_files = []

    # =============================================
    # Bloque de Entrenamiento
    # =============================================
    if not args.only_testing:
        # Configurar lista de archivos de entrenamiento
        train_files = sorted(
            [f for f in os.listdir(path_saludable) if f.endswith('.hdf5') and f != test_file],
            key=lambda x: re.search(r'(\d{2})_(\d{2})_(\d{4})', x).groups()[::-1]
        )
        logger.info(f"Archivos de entrenamiento encontrados: {len(train_files)}")

        # Fase 1: Recopilación de datos para normalización global
        logger.info("Fase 1/2: Recopilando datos para normalización global")
        all_train_data = []

        def load_and_preprocess(file_path):
            """Carga y preprocesa datos sin normalizar"""
            reader = HDF5Reader(file_path, json_file)
            timestamps = reader.print_timestamps(dataset_name, print_timestamps=False)
            df = reader.load_all_signals_for_timestamps(dataset_name, timestamps)
            if df.empty:
                return None
            return df.drop(columns=['Time', 'Timestamp'], errors='ignore').values.astype(np.float32)

        # Cargar todos los datos de entrenamiento
        for train_file in tqdm(train_files, desc="Cargando datos crudos"):
            file_path = os.path.join(path_saludable, train_file)
            data = load_and_preprocess(file_path)
            if data is not None:
                all_train_data.append(data)

        if not all_train_data:
            raise ValueError("No hay datos válidos para entrenamiento")

        # Ajustar scaler con todos los datos
        full_train_data = np.concatenate(all_train_data)
        scaler.fit(full_train_data)
        logger.info(f"Scaler ajustado con {len(full_train_data)} muestras")

        # Guardar scaler y metadatos
        joblib.dump(scaler, 'output/scaler.save')
        with open('output/n_features.txt', 'w') as f:
            f.write(str(full_train_data.shape[1]))
        logger.info(f"Características guardadas: {full_train_data.shape[1]}")

        # Liberar memoria
        del all_train_data, full_train_data
        gc.collect()

        # Fase 2: Procesamiento de ventanas con scaler global
        logger.info("Fase 2/2: Procesando ventanas de entrenamiento")
        for train_file in tqdm(train_files, desc="Generando ventanas"):
            file_path = os.path.join(path_saludable, train_file)
            reader = HDF5Reader(file_path, json_file)
            timestamps = reader.print_timestamps(dataset_name, print_timestamps=False)
            df = reader.load_all_signals_for_timestamps(dataset_name, timestamps)
            
            if df.empty:
                continue

            # Procesar con scaler global
            data = scaler.transform(df.drop(columns=['Time', 'Timestamp'], errors='ignore')).astype(np.float32)
            windows = create_windows(data, window_size, step_size)
            
            if windows is not None:
                all_windows.append(windows)

        # Validar datos
        if not all_windows:
            raise ValueError("No se generaron ventanas válidas")
        
        combined_windows = np.concatenate(all_windows, axis=0)
        logger.info(f"Total de ventanas de entrenamiento: {combined_windows.shape[0]}")

        # División de datasets
        train_windows, temp_windows = train_test_split(combined_windows, test_size=0.2, random_state=42)
        calibration_windows, val_windows = train_test_split(temp_windows, test_size=0.5, random_state=42)
        logger.info(f"División de datos - Train: {train_windows.shape[0]}, Val: {val_windows.shape[0]}, Calib: {calibration_windows.shape[0]}")

        # Guardar datos de calibración
        np.save('output/calibration_data.npy', calibration_windows)
        logger.info("Datos de calibración guardados")

        # Entrenar modelo
        input_size = train_windows.shape[1]
        model = AutoencoderKAN(input_size).to(device)
        logger.info(f"Arquitectura del modelo:\n{model}")

        auto_batch_size = GPUMemoryManager.auto_batch_size(device, model, input_size)
        logger.info(f"Batch size automático calculado: {auto_batch_size}")

        trained_model, _, val_loss_history = train_autoencoder(
            model,
            train_windows,
            val_data=val_windows,
            epochs=100,
            learning_rate=0.001,
            batch_size=auto_batch_size,
            patience=3,
            device=device
        )

        # Guardar modelo
        best_epoch = np.argmin(val_loss_history) + 1
        model_name = f"best_model_ep{best_epoch}_val{val_loss_history[best_epoch-1]:.6f}.pth"
        torch.save(trained_model.state_dict(), os.path.join('output', model_name))
        logger.info(f"Modelo guardado: {model_name}")

    # =============================================
    # Bloque de Inferencia
    # =============================================
    try:
        # Cargar recursos necesarios
        scaler = joblib.load('output/scaler.save')
        with open('output/n_features.txt', 'r') as f:
            n_features = int(f.read().strip())
        
        model_files = [f for f in os.listdir('output') if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError("No se encontraron modelos entrenados")
        
        logger.info(f"Cargando modelo: {model_files[0]}")
        model = AutoencoderKAN(400 * n_features).to(device)
        model.load_state_dict(torch.load(os.path.join('output', model_files[0]), weights_only=True))
        model.eval()
        
        # Configurar detector
        manual_predictor = ManualADPredictor(model)
        cad = ConformalAnomalyDetector(manual_predictor, batch_size=10)
        
        # Cargar datos de calibración
        if args.only_testing:
            logger.info("Cargando datos de calibración pre-guardados")
            calibration_windows = np.load('output/calibration_data.npy')
        else:
            logger.info("Usando datos de calibración recién generados")
        
        q_hat = cad.fit(calibration_windows, alpha=0.05)
        logger.info(f"Umbral de anomalía calculado: {q_hat:.4f}")

        # Procesar datos de test
        test_reader = HDF5Reader(os.path.join(path_saludable, test_file), json_file)
        test_timestamps = test_reader.print_timestamps(dataset_name, print_timestamps=False)
        logger.info(f"Total de timestamps a procesar: {len(test_timestamps)}")

        BATCH_SIZE = 10
        total_batches = len(test_timestamps) // BATCH_SIZE + 1
        
        with tqdm(total=total_batches, desc="Procesando test") as progress_bar:
            for batch_idx, test_start in enumerate(range(0, len(test_timestamps), BATCH_SIZE)):
                test_end = min(test_start + BATCH_SIZE, len(test_timestamps))
                batch_timestamps = test_timestamps[test_start:test_end]
                
                df_batch = test_reader.load_all_signals_for_timestamps(dataset_name, batch_timestamps)
                
                # Procesar timestamps
                if not df_batch.empty:
                    try:
                        df_batch['Timestamp'] = pd.to_datetime(
                            df_batch['Timestamp'],
                            format='%H_%M_%S',
                            errors='coerce'
                        )
                        df_batch = df_batch.dropna(subset=['Timestamp'])
                    except Exception as e:
                        logger.error(f"Error procesando timestamps: {str(e)}")
                        continue
                
                if df_batch.empty:
                    logger.warning(f"Batch {batch_idx+1} vacío")
                    progress_bar.update(1)
                    continue
                
                try:
                    with torch.no_grad():
                        timestamps = df_batch['Timestamp'].copy()
                        data = scaler.transform(df_batch.drop(columns=['Time', 'Timestamp'], errors='ignore')).astype(np.float32)
                        windows = create_windows(data, window_size, step_size)
                        
                        if windows is None or windows.size == 0:
                            logger.warning(f"No se generaron ventanas en batch {batch_idx+1}")
                            continue
                        
                        # Procesar en sub-lotes
                        sub_batch_size = 5
                        reconstruction = []
                        for i in range(0, len(windows), sub_batch_size):
                            sub_windows = windows[i:i+sub_batch_size]
                            tensor_windows = torch.from_numpy(sub_windows).to(device)
                            
                            sub_reconstruction = manual_predictor.predict(tensor_windows.cpu().numpy())
                            reconstruction.extend(sub_reconstruction)
                            
                            del tensor_windows, sub_reconstruction
                            torch.cuda.empty_cache()
                        
                        anomalies = (np.array(reconstruction) > q_hat).astype(int)

                        # Generar resultados
                        windowed_ts = pd.to_datetime(
                            pd.Series(create_windowed_timestamps(timestamps, window_size, step_size))
                        )
                        
                        output_df = pd.DataFrame({
                            'Timestamp': windowed_ts,
                            'Error_Reconstrucción': reconstruction,
                            'Anomalia': anomalies
                        })
                        output_path = f'output/test_batch_{batch_idx+1:03d}.csv'
                        output_df.to_csv(output_path, index=False)
                        logger.info(f"Resultados guardados en {output_path}")

                        # Generar gráfico
                        plt.figure(figsize=(12, 6))
                        plt.plot(reconstruction, label='Error de reconstrucción')
                        plt.axhline(q_hat, color='r', linestyle='--', label='Umbral de anomalía')
                        plt.title(f'Batch {batch_idx+1} - {windowed_ts.iloc[0]} a {windowed_ts.iloc[-1]}')
                        plt.xticks(
                            ticks=range(len(windowed_ts)),
                            labels=windowed_ts.dt.strftime('%Y-%m-%d %H:%M'),
                            rotation=45
                        )
                        plt.tight_layout()
                        plot_path = f'output/plots/batch_{batch_idx+1:03d}.png'
                        plt.savefig(plot_path)
                        plt.close()
                        logger.info(f"Gráfico guardado en {plot_path}")

                except Exception as batch_error:
                    logger.error(f"Error en batch {batch_idx+1}: {str(batch_error)}", exc_info=True)
                
                finally:
                    # Limpieza segura
                    vars_to_delete = ['df_batch', 'data', 'windows', 'reconstruction', 'anomalies']
                    for var in vars_to_delete:
                        if var in locals():
                            del locals()[var]
                    gc.collect()
                    progress_bar.update(1)

    except Exception as e:
        logger.error(f"Error en fase de inferencia: {str(e)}", exc_info=True)
        raise

    # Finalización
    exec_time = datetime.now() - start_time
    logger.info(f"Proceso completado. Tiempo total: {exec_time}")
    logger.info("Resultados disponibles en: output/")

if __name__ == "__main__":
    main()