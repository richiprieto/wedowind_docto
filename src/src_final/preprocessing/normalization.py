import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump

def normalize_dataset(dataset, method='minmax', scaler_filename=None, exclude_columns=['timestamp', 'date']):
    """
    Normaliza un dataset excluyendo columnas específicas por defecto
    
    Args:
        dataset (pd.DataFrame o tuple): DataFrame a normalizar o tupla conteniendo el DataFrame
        method (str): Método de normalización ('minmax' o 'standard')
        scaler_filename (str): Nombre para guardar el escalador (opcional)
        exclude_columns (list): Columnas a excluir de la normalización
    
    Returns:
        pd.DataFrame: DataFrame normalizado
    """
    # Crear directorio output si no existe
    os.makedirs('output', exist_ok=True)
    
    # Si es una tupla, tomar el primer elemento (asumiendo que es el DataFrame principal)
    if isinstance(dataset, tuple):
        data_to_process = dataset[0]
    else:
        data_to_process = dataset.copy()
    
    # Separar columnas a excluir
    existing_exclude = [col for col in exclude_columns if col in data_to_process.columns]
    features_to_normalize = data_to_process.drop(columns=existing_exclude, errors='ignore')
    excluded_data = data_to_process[existing_exclude] if existing_exclude else pd.DataFrame()
    
    # Seleccionar el método de normalización
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Método no válido. Usar 'minmax' o 'standard'")
    
    # Ajustar y transformar los datos
    normalized_data = scaler.fit_transform(features_to_normalize)
    
    # Reconstruir DataFrame
    normalized_df = pd.DataFrame(normalized_data,
                                columns=features_to_normalize.columns,
                                index=data_to_process.index)
    
    # Combinar con columnas excluidas
    if not excluded_data.empty:
        normalized_df = pd.concat([excluded_data, normalized_df], axis=1)
        normalized_df = normalized_df[data_to_process.columns]  # Mantener orden original
    
    # Guardar el escalador
    if not scaler_filename:
        scaler_filename = f'output/{method}_scaler.pkl'
    else:
        scaler_filename = f'output/{scaler_filename}'
    
    dump(scaler, scaler_filename)
    
    return normalized_df
