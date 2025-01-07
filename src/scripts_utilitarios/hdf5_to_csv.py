import h5py
import fireducks.pandas as pd
import glob
import os
from datetime import datetime
import numpy as np

def parse_timestamp(date_str, time_str):
    """
    Convierte la fecha del archivo y el timestamp en un datetime
    date_str formato: "06_02_2022"
    time_str formato: "00_13_12"
    """
    day, month, year = map(int, date_str.split('_'))
    hour, minute, second = map(int, time_str.split('_'))
    return datetime(year, month, day, hour, minute, second)

def convert_hdf5_to_feather(input_dir, min_samples=120000):
    """
    Convierte todos los archivos HDF5 en el directorio a un archivo Feather.
    Crea un archivo Feather por cada archivo HDF5.
    Solo incluye sensores con más de min_samples datos.
    Ignora sensores que comiencen con "Channel".
    """
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    
    for hdf5_path in hdf5_files:
        print(f"\nProcesando archivo: {hdf5_path}")
        
        # Extraer la fecha del nombre del archivo
        filename = os.path.basename(hdf5_path)  # Aventa_Taggenberg_06_02_2022.hdf5
        date_part = filename.replace('.hdf5', '').split('_')[-3:]  # ['06', '02', '2022']
        date_str = '_'.join(date_part)  # '06_02_2022'
        print(f"Fecha extraída del archivo: {date_str}")
        
        # Lista para almacenar todos los DataFrames de este archivo
        all_dfs = []
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'Aventa' not in f:
                print(f"No se encontró la carpeta 'Aventa' en {hdf5_path}.")
                continue
                
            timestamps = list(f['Aventa'].keys())
            
            for tstamp in timestamps:
                sensors = list(f[f'Aventa/{tstamp}'].keys())
                # Ignorar sensores que comiencen con "Channel"
                sensors = [s for s in sensors if not s.startswith("Channel")]
                
                if not sensors:
                    continue
                
                # Filtrar los sensores que tengan al menos 'min_samples'
                valid_sensors = []
                for sensor in sensors:
                    value_path = f'Aventa/{tstamp}/{sensor}/Value'
                    if value_path in f and len(f[value_path]) >= min_samples:
                        valid_sensors.append(sensor)
                
                if not valid_sensors:
                    print(f"No se encontraron sensores con más de {min_samples} muestras en timestamp {tstamp}")
                    continue
                
                # Recolectar datos de cada sensor válido
                sensor_data = {}
                max_length = 0
                for sensor in valid_sensors:
                    time_path = f'Aventa/{tstamp}/{sensor}/Time'
                    value_path = f'Aventa/{tstamp}/{sensor}/Value'
                    if time_path in f and value_path in f:
                        times = np.array(f[time_path][:], dtype=float).ravel()
                        values = np.array(f[value_path][:], dtype=float).ravel()
                        
                        sensor_data[sensor] = {
                            'times': times,
                            'values': values
                        }
                        max_length = max(max_length, len(times))
                
                if not sensor_data:
                    continue
                
                # Crear DataFrame para este timestamp
                data_dict = {
                    'timestamp': [],
                    'fecha': []
                }
                
                # Inicializar columnas para cada sensor
                for sensor in sensor_data.keys():
                    data_dict[sensor] = []
                
                # Obtener el datetime base para este timestamp
                base_datetime = parse_timestamp(date_str, tstamp)
                
                # Llenar el DataFrame
                for i in range(max_length):
                    data_dict['timestamp'].append(tstamp)
                    data_dict['fecha'].append(base_datetime.strftime('%Y-%m-%d'))
                    
                    for sensor, data in sensor_data.items():
                        if i < len(data['values']):
                            data_dict[sensor].append(float(data['values'][i]))
                        else:
                            data_dict[sensor].append(None)
                
                # Crear DataFrame para este timestamp y agregarlo a la lista
                print(tstamp)
                df_timestamp = pd.DataFrame(data_dict)
                all_dfs.append(df_timestamp)
        
        if not all_dfs:
            print(f"No se encontraron datos válidos en: {hdf5_path}")
            continue
        
        # Combinar todos los DataFrames de este archivo
        df_final = pd.concat(all_dfs, ignore_index=True)
        
        # Ordenar columnas
        columns = ['timestamp', 'fecha'] + [col for col in df_final.columns if col not in ['timestamp', 'fecha']]
        df_final = df_final[columns]
        
        # Generar el nombre de salida en formato Feather
        feather_path = os.path.splitext(hdf5_path)[0] + '.feather'
        
        # Guardar en formato Feather
        # Notar que se requiere 'pyarrow' para to_feather
        df_final.to_feather(feather_path)
        print(f"\nFeather guardado en: {feather_path}")
        print(f"Número total de filas procesadas: {len(df_final)}")
        print(f"Sensores incluidos: {[col for col in df_final.columns if col not in ['timestamp', 'fecha']]}")

if __name__ == "__main__":
    input_directory = "../../../primer_modelo"
    convert_hdf5_to_feather(input_directory, min_samples=120000) 