import h5py
import pandas as pd
import numpy as np
import json
from scipy.spatial.transform import Rotation


class HDF5Reader:
    def __init__(self, hdf5_file, json_file):
        """
        Inicializa el lector de archivos HDF5 y JSON.

        :param hdf5_file: Ruta al archivo HDF5.
        :param json_file: Ruta al archivo JSON con las especificaciones de los sensores.
        """
        self.hdf5_file = hdf5_file
        self.json_file = json_file
        self.sensors_specs = self.load_sensors_specs()

    def load_sensors_specs(self):
        """
        Carga las especificaciones de los sensores desde un archivo JSON.

        :return: Diccionario con los datos de los sensores.
        """
        with open(self.json_file, "r") as file:
            sensors_specs = json.load(file)
        return sensors_specs

    def get_generator_orientation(self, sensor_id="GEN_01"):
        """
        Obtiene la orientación (yaw, pitch, roll) de un sensor del generador desde las especificaciones.

        :param sensor_id: ID del sensor del generador.
        :return: Lista de ángulos de yaw, pitch y roll.
        """
        for sensor in self.sensors_specs["sensors"]:
            if sensor["sensor_placement"]["id"] == sensor_id:
                return sensor["sensor_placement"]["yaw-pitch-roll"]
        return None

    def rotate_generator_data(
        self, sensor_data, yaw_pitch_roll, sensor_name_prefix="GEN_ACC"
    ):
        """
        Rota los datos de un sensor del generador según una orientación dada (yaw, pitch, roll).

        :param sensor_data: Diccionario que contiene los datos de los sensores.
        :param yaw_pitch_roll: Lista o array con los ángulos de yaw, pitch, y roll en grados.
        :param sensor_name_prefix: Prefijo del nombre del sensor (por ejemplo: "GEN_ACC").
        :return: Diccionario con los datos rotados.
        """
        # Nombres de los sensores del generador
        try:
            Xs = sensor_data[f"{sensor_name_prefix}_XX_01_Value"]
            Ys = sensor_data[f"{sensor_name_prefix}_YY_01_Value"]
            Zs = sensor_data[f"{sensor_name_prefix}_ZZ_01_Value"]
        except KeyError:
            print(
                f"Error: No se encontraron las claves para {sensor_name_prefix}_XX_01, {sensor_name_prefix}_YY_01, {sensor_name_prefix}_ZZ_01"
            )
            return sensor_data

        # Crear un array con los datos originales de aceleración
        accelerometer_data = np.vstack(
            [Xs, Ys, Zs]
        ).T  # Cada fila es una muestra [X, Y, Z]

        # Aplicar la rotación a los datos de aceleración
        rotation = Rotation.from_euler("ZYX", yaw_pitch_roll, degrees=True)
        rotated_accelerometer_data = rotation.apply(accelerometer_data)

        # Asignar los datos rotados a las nuevas columnas
        sensor_data[f"{sensor_name_prefix}_Xt"] = rotated_accelerometer_data[:, 0]
        sensor_data[f"{sensor_name_prefix}_Yt"] = rotated_accelerometer_data[:, 1]
        sensor_data[f"{sensor_name_prefix}_Zt"] = rotated_accelerometer_data[:, 2]

        # Eliminar las columnas originales (opcional)
        del sensor_data[f"{sensor_name_prefix}_XX_01_Value"]
        del sensor_data[f"{sensor_name_prefix}_YY_01_Value"]
        del sensor_data[f"{sensor_name_prefix}_ZZ_01_Value"]

        return sensor_data

    def load_all_signals_for_timestamp(self, dataset_name, time_stamp):
        """
        Carga los datos de todos los sensores para un timestamp específico,
        ignorando aquellos sensores que tengan menos de 120,000 datos.
        Solo se guarda una columna de Time, compartida para todos los sensores.
        Rota los datos del sensor del generador.

        :param dataset_name: Nombre del dataset principal en el archivo HDF5.
        :param time_stamp: Marca de tiempo para la cual cargar los datos.
        :return: DataFrame con los datos de los sensores para el timestamp dado.
        """
        with h5py.File(self.hdf5_file, "r") as f:
            dataset = f[dataset_name][time_stamp]
            sensor_data = {}
            time_data_shared = None  # Para almacenar la primera columna de Time

            # Imprimir todas las claves de los sensores para verificar los nombres correctos
            print("Sensores disponibles:")
            for signal_name in dataset.keys():
                print(signal_name)

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
                            sensor_data["Time"] = (
                                time_data_shared  # Añadirla al DataFrame
                            )

                        # Aplanar las matrices de valores y almacenarlas
                        sensor_data[f"{signal_name}_Value"] = value_data.flatten()

            # Obtener la orientación del sensor del generador desde el archivo JSON
            yaw_pitch_roll = self.get_generator_orientation(sensor_id="GEN_01")
            if yaw_pitch_roll:
                sensor_name_prefix = "GEN_ACC"
                sensor_data = self.rotate_generator_data(
                    sensor_data, yaw_pitch_roll, sensor_name_prefix
                )
            else:
                print("No se encontró la orientación para el sensor del generador.")

            # Convertir a DataFrame
            df = pd.DataFrame(sensor_data)

        return df

    def print_timestamps(self, dataset_name):
        """
        Imprime todos los timestamps disponibles en el dataset especificado.
        
        :param dataset_name: Nombre del dataset principal en el archivo HDF5.
        """
        with h5py.File(self.hdf5_file, "r") as f:
            dataset = f[dataset_name]
            print("Timestamps disponibles:")
            for timestamp in dataset.keys():
                print(timestamp)
