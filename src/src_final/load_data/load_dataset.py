import h5py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_file(file_path, dataset_name='Aventa', min_size=120000, high_verbosity=False):
    """
    Procesa un único archivo HDF5 y retorna un DataFrame concatenado.
    
    Parámetros:
        file_path (str): Ruta al archivo HDF5.
        dataset_name (str): Nombre del dataset a extraer.
        min_size (int): Tamaño mínimo de datos requeridos por sensor.
        high_verbosity (bool): Si es True, se muestra logging detallado.
    
    Retorna:
        pd.DataFrame: DataFrame con la información concatenada, incluyendo las columnas
                      'date' (extraída del nombre del archivo) y 'timestamp' (de cada grupo).
    """
    dfs = []
    # Parsear la fecha a partir del nombre del archivo
    filename = os.path.basename(file_path)
    # Se espera que el nombre tenga el formato: <algo>_<algo>_DD_MM_YYYY.hdf5
    date_str = '_'.join(filename.split('_')[-3:]).split('.')[0]
    base_date = datetime.strptime(date_str, '%d_%m_%Y')
    if high_verbosity:
        logging.info(f"Procesando archivo: {file_path} con fecha base: {base_date}")
    else:
        logging.info(f"Procesando archivo: {file_path}")
    
    with h5py.File(file_path, "r") as f:
        # Obtener la lista de timestamps disponibles en el dataset
        time_names = list(f[dataset_name].keys())
        if high_verbosity:
            logging.info(f"Timestamps encontrados: {time_names}")
        
        # Iterar sobre cada timestamp
        for time_stamp in time_names:
            if high_verbosity:
                logging.info(f"Procesando timestamp: {time_stamp}")
            data = {}
            # Filtrar los sensores que no comienzan con "Channel"
            sensor_names = [sensor for sensor in f[dataset_name][time_stamp].keys() if not sensor.startswith("Channel")]
            for sensor in sensor_names:
                # Leer los valores, aplanarlos y convertirlos a float32 para ahorrar memoria
                sensor_values = f[dataset_name][time_stamp][sensor]['Value'][()]
                sensor_values = sensor_values.flatten().astype(np.float32)
                if sensor_values.size > min_size:
                    data[sensor] = sensor_values
                    if high_verbosity:
                        logging.info(f"Sensor {sensor} procesado con {sensor_values.size} datos")
            if data:
                df_temp = pd.DataFrame(data)
                # Insertar la fecha y el timestamp en columnas
                df_temp.insert(0, "timestamp", time_stamp)
                base_date_np = np.datetime64(base_date, 'D')
                df_temp.insert(0, "date", base_date_np)
                # Convertir la columna timestamp a tipo categórico para optimizar memoria
                df_temp["timestamp"] = df_temp["timestamp"].astype("category")
                dfs.append(df_temp)
                if high_verbosity:
                    logging.info(f"DataFrame para timestamp {time_stamp} creado con forma: {df_temp.shape}")
            else:
                if high_verbosity:
                    logging.info(f"No se encontraron datos suficientes para timestamp: {time_stamp}")
    
    if dfs:
        df_result = pd.concat(dfs, axis=0, ignore_index=True)
        logging.info(f"Archivo {file_path} procesado. DataFrame final tiene forma: {df_result.shape}")
        return df_result
    else:
        logging.info(f"Archivo {file_path} procesado sin datos válidos.")
        return pd.DataFrame()

def process_multiple_files(file_list, dataset_name='Aventa', min_size=120000, high_verbosity=False):
    """
    Procesa una lista de archivos HDF5 y concatena los DataFrames resultantes en uno solo.
    
    Parámetros:
        file_list (list): Lista de rutas a archivos HDF5.
        dataset_name (str): Nombre del dataset a extraer.
        min_size (int): Tamaño mínimo de datos requeridos por sensor.
        high_verbosity (bool): Si es True, se muestra logging detallado.
    
    Retorna:
        pd.DataFrame: DataFrame concatenado con la información de todos los archivos.
    """
    all_dfs = []
    for file_path in file_list:
        logging.info(f"Iniciando procesamiento del archivo: {file_path}")
        df = process_file(file_path, dataset_name, min_size, high_verbosity)
        if not df.empty:
            all_dfs.append(df)
        else:
            logging.info(f"No se obtuvieron datos válidos de {file_path}")
    if all_dfs:
        df_final = pd.concat(all_dfs, axis=0, ignore_index=True)
        logging.info(f"Procesamiento completo. DataFrame final tiene forma: {df_final.shape}")
        return df_final
    else:
        logging.info("No se procesó ningún archivo con datos válidos.")
        return pd.DataFrame()

# Ejemplo de uso:
#file_list = [
#    "../../aventa_rotor_icing/Aventa_Taggenberg_01_11_2022.hdf5",
#    "../../aventa_rotor_icing/Aventa_Taggenberg_04_11_2022.hdf5"
#]

# Para baja verbosidad (default):
# df_total = process_multiple_files(file_list)
# print(df_total)

# Para alta verbosidad:
# df_total = process_multiple_files(file_list, high_verbosity=True)
# print(df_total)
