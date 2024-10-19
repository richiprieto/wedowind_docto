# hdf5_reader.py
import h5py
import pandas as pd
import numpy as np
import json
from scipy.spatial.transform import Rotation


def rotate_generator_data(sensor_data, yaw_pitch_roll, sensor_name_prefix):
    """
    Rota los datos de un sensor del generador según una orientación dada (yaw, pitch, roll).

    :param sensor_data: Diccionario que contiene los datos de los sensores.
    :param yaw_pitch_roll: Lista o array con los ángulos de yaw, pitch, y roll en grados.
    :param sensor_name_prefix: Prefijo del nombre del sensor (por ejemplo: "GEN_ACC").
    :return: Diccionario con los datos rotados.
    """
    # Verificar si las claves XX, YY, ZZ existen en sensor_data
    try:
        Xs = sensor_data[f"{sensor_name_prefix}_XX_Value"]
        Ys = sensor_data[f"{sensor_name_prefix}_YY_Value"]
        Zs = sensor_data[f"{sensor_name_prefix}_ZZ_Value"]
    except KeyError:
        print(
            f"Error: No se encontraron las claves para {sensor_name_prefix}_XX, {sensor_name_prefix}_YY, {sensor_name_prefix}_ZZ"
        )
        return sensor_data

    # Crear un array con los datos originales de aceleración
    accelerometer_data = np.vstack([Xs, Ys, Zs]).T  # Cada fila es una muestra [X, Y, Z]

    # Aplicar la rotación a los datos de aceleración
    rotation = Rotation.from_euler("ZYX", yaw_pitch_roll, degrees=True)
    rotated_accelerometer_data = rotation.apply(accelerometer_data)

    # Asignar los datos rotados a las nuevas columnas
    sensor_data[f"{sensor_name_prefix}_Xt"] = rotated_accelerometer_data[:, 0]
    sensor_data[f"{sensor_name_prefix}_Yt"] = rotated_accelerometer_data[:, 1]
    sensor_data[f"{sensor_name_prefix}_Zt"] = rotated_accelerometer_data[:, 2]

    # Eliminar las columnas originales (opcional)
    del sensor_data[f"{sensor_name_prefix}_XX_Value"]
    del sensor_data[f"{sensor_name_prefix}_YY_Value"]
    del sensor_data[f"{sensor_name_prefix}_ZZ_Value"]

    return sensor_data


def load_sensors_specs(json_file):
    """
    Carga las especificaciones de los sensores desde un archivo JSON.

    :param json_file: Ruta al archivo JSON con las especificaciones de los sensores.
    :return: Diccionario con los datos de los sensores.
    """
    with open(json_file, "r") as file:
        sensors_specs = json.load(file)
    return sensors_specs


def get_generator_orientation(sensors_specs, sensor_id="GEN_01"):
    """
    Obtiene la orientación (yaw, pitch, roll) de un sensor del generador desde las especificaciones.

    :param sensors_specs: Diccionario con los datos de los sensores.
    :param sensor_id: ID del sensor del generador.
    :return: Lista de ángulos de yaw, pitch y roll.
    """
    for sensor in sensors_specs["sensors"]:
        if sensor["sensor_placement"]["id"] == sensor_id:
            return sensor["sensor_placement"]["yaw-pitch-roll"]
    return None


def load_all_signals_for_timestamp(file_path, dataset_name, time_stamp, sensors_specs):
    """
    Carga los datos de todos los sensores para un timestamp específico,
    ignorando aquellos sensores que tengan menos de 120,000 datos.
    Solo se guarda una columna de Time, compartida para todos los sensores.
    Rota los datos del sensor del generador.

    :param file_path: Ruta del archivo HDF5.
    :param dataset_name: Nombre del dataset principal en el archivo HDF5.
    :param time_stamp: Marca de tiempo para la cual cargar los datos.
    :param sensors_specs: Diccionario con las especificaciones de los sensores.
    :return: DataFrame con los datos de los sensores para el timestamp dado.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name][time_stamp]
        sensor_data = {}
        time_data_shared = None  # Para almacenar la primera columna de Time

        # Iterar sobre todos los sensores y cargar datos de Time y Value
        for signal_name in dataset.keys():
            if signal_name != "ChannelList":  # Evitar cargar la lista de canales
                time_data = dataset[signal_name]["Time"][()]
                value_data = dataset[signal_name]["Value"][()]

                # Ignorar los sensores que tienen menos de 120,000 datos
                if len(time_data) >= 120000 and len(value_data) >= 120000:
                    # Usar la primera columna de Time y aplicarla a todos los sensores
                    if time_data_shared is None:
                        time_data_shared = (
                            time_data.flatten()
                        )  # Guardar la primera columna de Time
                        sensor_data["Time"] = time_data_shared  # Añadirla al DataFrame

                    # Aplanar las matrices de valores y almacenarlas
                    sensor_data[f"{signal_name}_Value"] = value_data.flatten()

        # Obtener la orientación del sensor del generador desde el archivo JSON
        yaw_pitch_roll = get_generator_orientation(sensors_specs, sensor_id="GEN_01")
        if yaw_pitch_roll:
            sensor_name_prefix = "GEN_ACC"
            sensor_data = rotate_generator_data(
                sensor_data, yaw_pitch_roll, sensor_name_prefix
            )
        else:
            print("No se encontró la orientación para el sensor del generador.")

        # Convertir a DataFrame
        df = pd.DataFrame(sensor_data)

    return df


if __name__ == "__main__":
    # Ejemplo de uso
    file_path = "../dataset/Aventa_Taggenberg_16_02_2022.hdf5"
    json_file = "../dataset/Aventa_sensors.json"
    dataset_name = "Aventa"
    time_stamp = "23_53_12"

    # Cargar las especificaciones de los sensores desde el archivo JSON
    sensors_specs = load_sensors_specs(json_file)

    # Cargar los datos de todos los sensores para un timestamp específico
    df_sensors = load_all_signals_for_timestamp(
        file_path, dataset_name, time_stamp, sensors_specs
    )
    print(df_sensors.head())

    # Guardar el DataFrame a un archivo CSV si es necesario
    output_csv = f"{time_stamp}_sensors_data.csv"
    df_sensors.to_csv(output_csv, index=False)
    print(f"Datos guardados en: {output_csv}")
