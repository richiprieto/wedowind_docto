#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2024 Eón Corp

# Permiso se otorga gratuitamente a cualquier persona que obtenga una copia
# de este software y archivos de documentación asociados (el "Software"),
# para utilizarlo sin restricciones, incluyendo sin limitación los derechos
# para usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar,
# y/o vender copias del Software, y permitir a las personas a quienes se
# les proporcione el Software hacerlo, sujeto a las siguientes condiciones:

# El aviso de copyright anterior y este aviso de permiso se incluirán en
# todas las copias o partes sustanciales del Software.

# EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO,
# EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS
# DE COMERCIABILIDAD, IDONEIDAD PARA UN PROPÓSITO PARTICULAR Y NO
# INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DEL COPYRIGHT SERÁN
# RESPONSABLES DE NINGUNA RECLAMACIÓN, DAÑOS U OTRAS RESPONSABILIDADES,
# YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, QUE SURJAN
# DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTRO TIPO DE
# ACCIONES EN EL SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
import json
from read_hdf5 import HDF5Reader
import os

def cargar_dataset_saludable():
    """
    Carga el dataset saludable desde archivos HDF5 y JSON.

    Returns:
        datos (numpy.ndarray): Datos del dataset saludable.
        sensores (dict): Información de los sensores.
    """
    # Ruta al dataset saludable
    path_saludable = "../aventa_failure_flexible_coupling_of_collective_pitch_drive/"
    file_path_train = os.path.join(path_saludable, "Aventa_Taggenberg_15_02_2022.hdf5")
    json_file_train = os.path.join(path_saludable, "Aventa_sensors.json")
    dataset_name = "Aventa"

    reader_train = HDF5Reader(file_path_train, json_file_train)

    # Obtener los índices de corte basados en el tamaño del timestamp de entrenamiento
    train_timestamps = reader_train.print_timestamps(dataset_name)

    df_train = reader_train.load_all_signals_for_timestamps(dataset_name, train_timestamps)
    df_train = df_train.drop(columns=['Time', 'Timestamp'])

    return df_train

def main():
    # Cargar el dataset saludable
    datos = cargar_dataset_saludable()

    # Verificar que los datos no estén vacíos
    if datos.size == 0:
        raise ValueError("El dataset saludable está vacío.")

    # Estandarizar los datos
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(datos)
    print("Datos estandarizados.")

    # Aplicar PCA
    pca = PCA()
    pca.fit(X_scaled)
    print("PCA aplicado a los datos estandarizados.")

    # Obtener la varianza explicada por cada componente
    varianza = pca.explained_variance_ratio_
    print("Varianza explicada calculada.")

    # Calcular la varianza acumulada
    var_acum = np.cumsum(varianza)
    print("Varianza acumulada calculada.")

    # Visualizar la varianza explicada y la varianza acumulada
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(varianza) + 1), varianza, alpha=0.6, label='Varianza Explicada')
    plt.plot(range(1, len(var_acum) + 1), var_acum, marker='o', color='r', label='Varianza Acumulada')
    plt.title('Varianza Explicada y Acumulada por Componentes Principales - Dataset Saludable')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Proporción de Varianza')
    plt.xticks(range(1, len(varianza) + 1))
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% Varianza Acumulada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Gráfico de varianza explicada mostrado.")

if __name__ == "__main__":
    main()
