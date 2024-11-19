import os
import re
from datetime import datetime
import pandas as pd

def procesar_matriz_similitud(archivo_entrada, archivo_salida, directorio_raiz):
    """
    Procesa el archivo matriz_similitud.xlsx que contiene únicamente nombres de archivos .hdf5,
    extrae la información de la carpeta, ordena temporalmente y agrega columnas adicionales
    como los archivos duplicados y la fecha.

    :param archivo_entrada: Ruta del archivo Excel de entrada (matriz_similitud.xlsx).
    :param archivo_salida: Ruta del archivo Excel de salida (lista_datasets.xlsx).
    :param directorio_raiz: Ruta del directorio raíz donde se encuentran las carpetas que inician con 'aventa'.
    """
    # Expresiones regulares para identificar carpetas y archivos
    patron_carpeta = re.compile(r'^aventa', re.IGNORECASE)
    patron_archivo = re.compile(r'Aventa_[A-Za-z]+_(\d{2})_(\d{2})_(\d{4})\.hdf5$', re.IGNORECASE)
    
    # Leer el archivo Excel que contiene los nombres de los archivos
    df_similitud = pd.read_excel(archivo_entrada, header=None, names=['Archivo con Fecha'])
    
    datos = []

    # Recorrer cada archivo listado en matriz_similitud.xlsx
    for archivo in df_similitud['Archivo con Fecha']:
        match = patron_archivo.match(archivo)
        if match:
            dia, mes, año = match.groups()
            fecha = datetime.strptime(f"{dia}-{mes}-{año}", "%d-%m-%Y")
            
            # Buscar la carpeta que contiene el archivo
            carpeta_encontrada = None
            for carpeta in os.listdir(directorio_raiz):
                ruta_carpeta = os.path.join(directorio_raiz, carpeta)
                if os.path.isdir(ruta_carpeta) and patron_carpeta.match(carpeta):
                    ruta_archivo = os.path.join(ruta_carpeta, archivo)
                    if os.path.exists(ruta_archivo):
                        carpeta_encontrada = carpeta
                        break
            
            if carpeta_encontrada:
                datos.append({
                    'Nombre de la Carpeta': carpeta_encontrada,
                    'Archivo con Fecha': archivo,
                    'Fecha': fecha
                })
            else:
                print(f"Advertencia: Archivo '{archivo}' no encontrado en ninguna carpeta que inicie con 'aventa'.")

    # Crear un DataFrame de pandas
    df = pd.DataFrame(datos)
    
    if df.empty:
        print("No se encontraron archivos válidos para procesar.")
        return

    # Ordenar los datos por fecha
    datos_ordenados = sorted(datos, key=lambda x: x['Fecha'])

    df_ordenado = pd.DataFrame(datos_ordenados)
    
    # Identificar duplicados basados en 'Archivo con Fecha'
    df_ordenado['EsDuplicado'] = df_ordenado.duplicated(subset=['Archivo con Fecha'], keep=False)
    
    # Filtrar los duplicados
    duplicados = df_ordenado[df_ordenado['EsDuplicado']].copy()
    
    # Agrupar por 'Archivo con Fecha' y obtener los nombres de archivos duplicados
    duplicados['ArchivoDuplicado'] = duplicados.groupby('Archivo con Fecha')['Archivo con Fecha'].transform(lambda x: ', '.join(x.unique()))
    
    # Extraer solo la fecha en formato deseado
    duplicados['SoloFecha'] = duplicados['Fecha'].dt.strftime('%d-%m-%Y')
    
    # Eliminar duplicados dejando solo la primera aparición
    df_sin_duplicados = df_ordenado.drop_duplicates(subset=['Archivo con Fecha'], keep='first')
    
    # Combinar con la información de duplicados
    df_final = df_sin_duplicados.merge(
        duplicados[['Archivo con Fecha', 'ArchivoDuplicado', 'SoloFecha']],
        on='Archivo con Fecha',
        how='left'
    )
    
    # Rellenar los valores NaN para los archivos que no tienen duplicados
    df_final['ArchivoDuplicado'] = df_final['ArchivoDuplicado'].fillna('')
    df_final['SoloFecha'] = df_final['SoloFecha'].fillna(df_final['Fecha'].dt.strftime('%d-%m-%Y'))
    
    # Seleccionar las columnas finales
    columnas_finales = ['Nombre de la Carpeta', 'Archivo con Fecha', 'ArchivoDuplicado', 'SoloFecha']
    df_final = df_final[columnas_finales]
    
    # Guardar el resultado en un nuevo archivo Excel
    df_final.to_excel(archivo_salida, index=False)
    print(f"Archivo procesado y guardado en {archivo_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    archivo_entrada = 'matriz_similitud.xlsx'
    archivo_salida = 'lista_datasets.xlsx'
    directorio = '../../../'
    procesar_matriz_similitud(archivo_entrada, archivo_salida, directorio)
