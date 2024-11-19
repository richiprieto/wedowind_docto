#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2024 Eón Corp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def get_all_hdf5_files(root_dir, folder_prefix="aventa"):
    """
    Encuentra todos los archivos HDF5 en las carpetas que comienzan con el prefijo especificado.
    """
    hdf5_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).lower().startswith(folder_prefix):
            for file in filenames:
                if file.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(dirpath, file))
    return hdf5_files

def extract_date_from_filename(filename):
    """
    Extrae la fecha del nombre del archivo en formato Aventa_Taggenberg_dd_mm_yyyy.hdf5
    y la devuelve como un objeto datetime.
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) < 5:
            raise ValueError("Nombre de archivo no tiene el formato esperado 'Aventa_Taggenberg_dd_mm_yyyy.hdf5'.")
        # Unir las tres últimas partes para obtener 'dd_mm_yyyy'
        date_str = '_'.join(parts[-3:]).split('.')[0]  # Obtiene 'dd_mm_yyyy'
        return datetime.strptime(date_str, "%d_%m_%Y")
    except Exception as e:
        print(f"Error al extraer la fecha de {filename}: {e}")
        return datetime.min  # Retorna una fecha mínima en caso de error

def calculate_similarity(file1, file2, chunk_size=1024*1024):
    """
    Calcula el porcentaje de similitud bit a bit entre dos archivos HDF5.
    Devuelve un valor entre 0 y 1.
    """
    try:
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)

        if size1 != size2:
            # print(f"Los archivos {file1} y {file2} tienen tamaños diferentes.")
            return 0  # Archivos de diferentes tamaños, similitud 0

        if size1 == 0 and size2 == 0:
            return 1  # Ambos archivos están vacíos

        matching_bytes = 0
        total_bytes = size1

        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            while True:
                b1 = f1.read(chunk_size)
                b2 = f2.read(chunk_size)

                if not b1 and not b2:
                    break

                # Convertir a arrays de bytes para comparación
                arr1 = np.frombuffer(b1, dtype=np.uint8)
                arr2 = np.frombuffer(b2, dtype=np.uint8)

                matching_bytes += np.sum(arr1 == arr2)

        return matching_bytes / total_bytes
    except Exception as e:
        print(f"Error al calcular similitud entre {file1} y {file2}: {e}")
        return 0

def generate_similarity_matrix(hdf5_files):
    """
    Genera una matriz de similitud para una lista de archivos HDF5.
    """
    num_files = len(hdf5_files)
    similarity_matrix = np.zeros((num_files, num_files), dtype=float)

    # Establecer la diagonal en 1 ya que cada archivo es idéntico a sí mismo
    np.fill_diagonal(similarity_matrix, 1.0)

    print("Comparando archivos...")
    with tqdm(total=num_files * (num_files - 1) // 2) as pbar:  # Progreso para combinaciones únicas
        for i in range(num_files):
            for j in range(i + 1, num_files):  # Solo compara una vez cada par
                similarity = calculate_similarity(hdf5_files[i], hdf5_files[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Simetría
                pbar.update(1)

    return similarity_matrix

def verify_symmetry(matrix):
    """
    Verifica si la matriz es simétrica respecto a la diagonal principal.
    """
    if np.allclose(matrix, matrix.T):
        print("La matriz de similitud es simétrica.")
    else:
        print("Advertencia: La matriz de similitud NO es simétrica.")

def print_identical_files(matrix, files):
    """
    Imprime los pares de archivos que son exactamente iguales.
    """
    num_files = len(files)
    printed_pairs = set()
    print("\nPares de archivos idénticos (similitud = 1):")
    for i in range(num_files):
        for j in range(i + 1, num_files):
            if matrix[i][j] == 1.0:
                pair = tuple(sorted([files[i], files[j]]))
                if pair not in printed_pairs:
                    print(f"- {files[i]} <--> {files[j]}")
                    printed_pairs.add(pair)
    print("Fin de la lista de pares idénticos.\n")

def save_matrix_to_excel(matrix, files, output_file):
    """
    Guarda la matriz de similitud en un archivo Excel.
    """
    df = pd.DataFrame(matrix, index=files, columns=files)
    df.to_excel(output_file, index=True, header=True)
    print(f"Matriz de similitud guardada en: {output_file}")

def sort_files_by_date(files):
    """
    Ordena la lista de archivos HDF5 por la fecha extraída del nombre del archivo.
    """
    return sorted(files, key=extract_date_from_filename)

def main():
    # Directorio raíz donde buscar los archivos
    root_dir = "../../../"

    print("Buscando archivos HDF5...")
    hdf5_files = get_all_hdf5_files(root_dir)

    if not hdf5_files:
        print("No se encontraron archivos HDF5 en las carpetas que comienzan con 'aventa'.")
        return

    # Ordenar los archivos por fecha
    hdf5_files = sort_files_by_date(hdf5_files)
    print(f"Se encontraron {len(hdf5_files)} archivos HDF5 ordenados por fecha.")

    print("Generando la matriz de similitud...")
    similarity_matrix = generate_similarity_matrix(hdf5_files)

    print("Matriz de similitud generada con éxito.")

    # Verificar simetría de la matriz
    verify_symmetry(similarity_matrix)

    # Imprimir pares de archivos idénticos
    print_identical_files(similarity_matrix, hdf5_files)

    print("Guardando la matriz en un archivo Excel...")
    # Archivo de salida
    output_file = os.path.join(root_dir, "matriz_similitud.xlsx")
    save_matrix_to_excel(similarity_matrix, hdf5_files, output_file)

if __name__ == "__main__":
    main()
