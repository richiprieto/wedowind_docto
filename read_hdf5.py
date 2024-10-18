# read_hdf5.py
import h5py
import numpy as np


def load_hdf5_file(file_path, dataset_name):
    """
    Carga un dataset desde un archivo HDF5.

    :param file_path: Ruta del archivo HDF5.
    :param dataset_name: Nombre del dataset dentro del archivo HDF5.
    :return: Datos cargados como numpy array.
    """
    with h5py.File(file_path, "r") as f:
        data = f[dataset_name][:]
    return np.array(data)


if __name__ == "__main__":
    # Ejemplo de uso
    file_path = "data/your_dataset.h5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)
    print(f"Datos cargados desde {file_path} con tamaño {data.shape}")
