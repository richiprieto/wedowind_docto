import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
import pandas as pd

def comparar_archivos_hdf5(carpeta1, carpeta2):
    archivos1 = set(Path(carpeta1).glob('*.hdf5'))
    archivos2 = set(Path(carpeta2).glob('*.hdf5'))
    
    archivos_iguales = []
    archivos_diferentes = []

    # Crear barra de progreso
    barra_progreso = tqdm(total=len(archivos1), desc="Comparando archivos", unit="archivo")

    for archivo1 in archivos1:
        nombre_archivo = archivo1.name
        archivo2 = Path(carpeta2) / nombre_archivo
        
        if archivo2.exists():
            if son_archivos_identicos(archivo1, archivo2):
                archivos_iguales.append(nombre_archivo)
            else:
                archivos_diferentes.append((nombre_archivo, archivo1, archivo2))
        
        # Actualizar barra de progreso
        barra_progreso.update(1)
    
    # Cerrar barra de progreso
    barra_progreso.close()
    
    return archivos_iguales, archivos_diferentes

def son_archivos_identicos(archivo1, archivo2):
    with h5py.File(archivo1, 'r') as f1, h5py.File(archivo2, 'r') as f2:
        return comparar_grupos(f1, f2)

def comparar_grupos(grupo1, grupo2):
    if set(grupo1.keys()) != set(grupo2.keys()):
        return False
    
    for key in grupo1.keys():
        if isinstance(grupo1[key], h5py.Group):
            if not comparar_grupos(grupo1[key], grupo2[key]):
                return False
        elif isinstance(grupo1[key], h5py.Dataset):
            if not np.array_equal(grupo1[key][:], grupo2[key][:]):
                return False
    
    return True

def mostrar_diferencias(archivo1, archivo2):
    with h5py.File(archivo1, 'r') as f1, h5py.File(archivo2, 'r') as f2:
        return comparar_y_mostrar_diferencias(f1, f2)

def comparar_y_mostrar_diferencias(grupo1, grupo2, ruta=''):
    diferencias = []
    
    for key in set(grupo1.keys()) | set(grupo2.keys()):
        ruta_actual = f"{ruta}/{key}"
        
        if key not in grupo1:
            diferencias.append(f"El elemento '{ruta_actual}' existe en el segundo archivo pero no en el primero.")
        elif key not in grupo2:
            diferencias.append(f"El elemento '{ruta_actual}' existe en el primer archivo pero no en el segundo.")
        elif isinstance(grupo1[key], h5py.Group):
            diferencias.extend(comparar_y_mostrar_diferencias(grupo1[key], grupo2[key], ruta_actual))
        elif isinstance(grupo1[key], h5py.Dataset):
            if not np.array_equal(grupo1[key][:], grupo2[key][:]):
                diferencias.append(f"Los valores del dataset '{ruta_actual}' son diferentes.")
    
    return diferencias

def calcular_diferencia_datos(archivo1, archivo2):
    try:
        with h5py.File(archivo1, 'r') as f1, h5py.File(archivo2, 'r') as f2:
            diferencia_total = 0
            num_elementos = 0
            
            def comparar_grupos_recursivamente(grupo1, grupo2):
                nonlocal diferencia_total, num_elementos
                for key in grupo1.keys():
                    if key not in grupo2:
                        continue
                    if isinstance(grupo1[key], h5py.Dataset):
                        data1 = grupo1[key][:]
                        data2 = grupo2[key][:]
                        if data1.shape == data2.shape:
                            if np.issubdtype(data1.dtype, np.number) and np.issubdtype(data2.dtype, np.number):
                                diferencia = np.abs(data1 - data2)
                                diferencia_total += np.sum(diferencia)
                                num_elementos += np.prod(data1.shape)
                    elif isinstance(grupo1[key], h5py.Group):
                        comparar_grupos_recursivamente(grupo1[key], grupo2[key])
            
            comparar_grupos_recursivamente(f1, f2)
            
            if num_elementos > 0:
                return diferencia_total / num_elementos
            else:
                return -1  # Indicador de que no se encontraron elementos comparables
    except Exception as e:
        print(f"Error al comparar {archivo1} y {archivo2}: {str(e)}")
        return -2  # Indicador de error

def comparar_nombres_y_tamanos(carpeta1, carpeta2):
    archivos1 = list(Path(carpeta1).glob('*.hdf5'))
    archivos2 = list(Path(carpeta2).glob('*.hdf5'))
    
    resultados = []
    
    for archivo1 in tqdm(archivos1, desc="Comparando archivos", unit="archivo"):
        nombre1 = archivo1.name
        tamano1 = archivo1.stat().st_size
        
        for archivo2 in archivos2:
            nombre2 = archivo2.name
            tamano2 = archivo2.stat().st_size
            
            similitud_nombre = SequenceMatcher(None, nombre1, nombre2).ratio()
            diferencia_tamano = abs(tamano1 - tamano2)
            diferencia_datos = calcular_diferencia_datos(archivo1, archivo2)
            
            resultados.append({
                'Archivo1': nombre1,
                'Archivo2': nombre2,
                'Similitud_Nombre': similitud_nombre,
                'Diferencia_Tamaño': diferencia_tamano,
                'Tamaño1': tamano1,
                'Tamaño2': tamano2,
                'Diferencia_Datos': diferencia_datos
            })
    
    df = pd.DataFrame(resultados)
    
    # Ajustar la puntuación para manejar valores especiales de Diferencia_Datos
    df['Puntuación'] = df.apply(lambda row: 
        row['Similitud_Nombre'] - 
        row['Diferencia_Tamaño'] / df['Diferencia_Tamaño'].max() - 
        (0 if row['Diferencia_Datos'] < 0 else row['Diferencia_Datos'] / df[df['Diferencia_Datos'] >= 0]['Diferencia_Datos'].max()),
        axis=1
    )
    
    df = df.sort_values('Puntuación', ascending=False)
    
    return df

# Uso del código
carpeta1 = '../aventa_normal_operation_for_system_identification/'
carpeta2 = '../aventa_failure_flexible_coupling_of_collective_pitch_drive/'

print("Iniciando comparación de archivos...")
archivos_iguales, archivos_diferentes = comparar_archivos_hdf5(carpeta1, carpeta2)

print("\nComparando nombres, tamaños y datos de archivos...")
df_comparacion = comparar_nombres_y_tamanos(carpeta1, carpeta2)

print("\nTabla de comparación (ordenada de más similar a menos similar):")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df_comparacion.to_string(index=False))
df_comparacion.to_excel("tabla_comparada.xlsx")

print("\nArchivos exactamente iguales:")
for archivo in archivos_iguales:
    print(f"- {archivo}")

print("\nArchivos diferentes:")
for nombre, archivo1, archivo2 in archivos_diferentes:
    print(f"\nDiferencias en '{nombre}':")
    diferencias = mostrar_diferencias(archivo1, archivo2)
    for diferencia in diferencias:
        print(f"- {diferencia}")

print("\nProceso de comparación completado.")