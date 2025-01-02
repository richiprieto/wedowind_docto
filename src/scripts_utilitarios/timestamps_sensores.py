import os
from read_hdf5 import HDF5Reader

def procesar_archivo_hdf5(hdf5_path, json_path, debug=False):
    """
    Procesa un archivo HDF5 utilizando la clase HDF5Reader para extraer y mostrar
    los timestamps y los datos de los sensores.

    :param hdf5_path: Ruta al archivo HDF5.
    :param json_path: Ruta al archivo JSON con las especificaciones de los sensores.
    :param debug: Habilita el modo de depuración.
    """
    lector = HDF5Reader(hdf5_file=hdf5_path, json_file=json_path, debug=debug)
    dataset_name = "nombre_del_dataset"  # Reemplaza con el nombre real del dataset

    # Imprimir todos los timestamps disponibles
    timestamps = lector.print_timestamps(dataset_name)

    for timestamp in timestamps:
        df = lector.load_all_signals_for_timestamp(dataset_name, timestamp)
        if df.empty:
            continue
        print(f"Timestamp: {timestamp}")
        for column in df.columns:
            if column != "Time":
                print(f"  {column}: {df[column].values}")
        print()

def main():
    directorio_base = '.'  # Puedes cambiar esto al directorio raíz que desees
    debug_mode = False      # Cambia a True para habilitar mensajes de depuración

    for carpeta in os.listdir(directorio_base):
        if carpeta.startswith('aventa') and os.path.isdir(os.path.join(directorio_base, carpeta)):
            ruta_carpeta = os.path.join(directorio_base, carpeta)
            json_path = os.path.join(ruta_carpeta, 'aventa_sensors.json')
            hdf5_files = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.endswith('.hdf5')]

            if not os.path.exists(json_path):
                print(f"No se encontró el archivo {json_path} en la carpeta {ruta_carpeta}")
                continue

            if not hdf5_files:
                print(f"No se encontraron archivos .hdf5 en la carpeta {ruta_carpeta}")
                continue

            for archivo in hdf5_files:
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                print(f"Procesando archivo: {ruta_archivo}")
                procesar_archivo_hdf5(ruta_archivo, json_path, debug=debug_mode)

if __name__ == "__main__":
    main()
