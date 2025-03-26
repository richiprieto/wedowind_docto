import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def clean_dataset(df, handling_missing='drop', verbose=True):
    """
    Limpieza básica del dataset: manejo de valores faltantes
    """
    if verbose:
        logger.info("Iniciando limpieza básica del dataset")
    
    df = df.copy()
    
    # Manejo de valores faltantes
    missing = df.isnull()
    if missing.any().any():
        if verbose:
            total_missing = missing.sum().sum()
            logger.warning(f"Valores faltantes totales: {total_missing}")
            
        if handling_missing == 'drop':
            df = df.dropna()
        elif handling_missing == 'interpolate':
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].interpolate()
            df = df.ffill()
        elif handling_missing == 'ffill':
            df = df.ffill()
    else:
        if verbose:
            logger.info("No hay valores faltantes en el dataset")
    
    if verbose:
        logger.info("Limpieza básica completada. Dataset: %d filas x %d columnas", *df.shape)
    
    return df

def detect_and_handle_outliers(df, method='MAD', threshold=3.0, 
                              handling='clip', verbose=True):
    """
    Detección y manejo de outliers en columnas numéricas
    Retorna: DataFrame procesado y máscara de outliers
    """
    if verbose:
        logger.info("Iniciando detección de outliers")
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    outliers_mask = pd.Series(False, index=df.index)
    bounds = {}

    for col in numeric_cols:
        try:
            if method == 'MAD':
                median = df[col].median()
                mad = (df[col] - median).abs().median()
                if mad == 0:
                    if verbose:
                        logger.warning(f"Columna {col} tiene MAD=0, omitiendo")
                    continue
                z_scores = 0.6745 * (df[col] - median) / mad
                lower = median - threshold * mad / 0.6745
                upper = median + threshold * mad / 0.6745
                
            elif method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    if verbose:
                        logger.warning(f"Columna {col} tiene IQR=0, omitiendo")
                    continue
                z_scores = (df[col] - df[col].median()) / IQR
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
            
            col_outliers = (z_scores.abs() > threshold)
            outliers_mask |= col_outliers
            bounds[col] = (lower, upper)
            
            if verbose and col_outliers.any():
                logger.warning(f"Columna {col}: {col_outliers.sum()} outliers")

        except Exception as e:
            logger.error(f"Error procesando {col}: {str(e)}")
            continue

    # Aplicar manejo de outliers
    if handling == 'remove':
        df = df[~outliers_mask]
    elif handling == 'clip':
        for col, (lower, upper) in bounds.items():
            df[col] = df[col].clip(lower=lower, upper=upper)
    elif handling == 'log':
        for col in numeric_cols:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    
    if verbose:
        logger.info(f"Outliers manejados ({handling}): {outliers_mask.sum()} registros afectados")
    
    return df, outliers_mask
