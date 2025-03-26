import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def clean_dataset(df, handling_missing='drop', 
                 detect_outliers=True, outlier_method='MAD', threshold=3.0,
                 handling_outliers='clip', verbose=True):
    """
    Limpieza general de dataset con detección de outliers
    """
    if verbose:
        logger.info("Iniciando limpieza completa del dataset")
    
    df = df.copy()
    
    # Manejo de valores faltantes en todas las columnas
    missing = df.isnull()
    if missing.any().any():
        if verbose:
            total_missing = missing.sum().sum()
            logger.warning(f"Valores faltantes totales: {total_missing}")
            
        if handling_missing == 'drop':
            df = df.dropna()
        elif handling_missing == 'interpolate':
            # Interpola solo columnas numéricas
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].interpolate()
            df = df.ffill()  # Para columnas no numéricas
        elif handling_missing == 'ffill':
            df = df.ffill()
    else:
        logger.info("No hay valores faltantes en el dataset")
    
    # Nueva sección para manejo de outliers
    if detect_outliers:
        if verbose:
            logger.info("Iniciando detección de valores atípicos")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        outliers_mask = pd.Series(False, index=df.index)
        
        for col in numeric_cols:
            if outlier_method == 'MAD':
                median = df[col].median()
                mad = (df[col] - median).abs().median()
                if mad == 0:  # Evitar división por cero
                    if verbose:
                        logger.warning(f"Columna {col} tiene MAD=0, omitiendo detección")
                    continue
                z_scores = 0.6745 * (df[col] - median) / mad
            elif outlier_method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    if verbose:
                        logger.warning(f"Columna {col} tiene IQR=0, omitiendo detección")
                    continue
                z_scores = (df[col] - df[col].median()) / IQR
            
            col_outliers = (z_scores.abs() > threshold)
            outliers_mask |= col_outliers
            
            if verbose and col_outliers.any():
                logger.warning(f"Columna {col}: {col_outliers.sum()} outliers detectados")
                
        if handling_outliers == 'remove':
            df = df[~outliers_mask]
        elif handling_outliers == 'clip':
            for col in numeric_cols:
                if outlier_method == 'MAD':
                    median = df[col].median()
                    mad = (df[col] - median).abs().median()
                    lower = median - threshold * mad / 0.6745
                    upper = median + threshold * mad / 0.6745
                elif outlier_method == 'IQR':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                
                df[col] = df[col].clip(lower=lower, upper=upper)
        elif handling_outliers == 'log':
            for col in numeric_cols:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        
        if verbose:
            logger.info(f"Outliers manejados: {outliers_mask.sum()} registros afectados")
    
    if verbose:
        logger.info("Limpieza completada. Dataset final: %d filas x %d columnas", *df.shape)
    
    return df
