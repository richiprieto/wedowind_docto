import numpy as np
import pandas as pd

def generate_normal_signal(duration, sample_rate):
    num_samples = int(duration * sample_rate)
    
    # Generar timestamp
    timestamp = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generar señales aleatorias para cada eje
    x = np.random.normal(0, 1, num_samples)
    y = np.random.normal(0, 1, num_samples)
    z = np.random.normal(0, 1, num_samples)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Time': timestamp,
        'x': x,
        'y': y,
        'z': z
    })
    
    return df

def generate_anomalous_signal(duration, sample_rate, anomaly_percentage):
    # Generar señal normal
    df = generate_normal_signal(duration, sample_rate)
    
    # Calcular la duración de la anomalía basada en el porcentaje
    anomaly_duration = duration * (anomaly_percentage / 100)
    
    # Calcular el inicio aleatorio de la anomalía
    max_start_time = duration - anomaly_duration
    anomaly_start = np.random.uniform(0, max_start_time)
    
    # Calcular índices de inicio y fin de la anomalía
    start_index = int(anomaly_start * sample_rate)
    end_index = start_index + int(anomaly_duration * sample_rate)
    
    # Introducir anomalía en cada eje
    for axis in ['x', 'y', 'z']:
        # Amplificar la señal durante la anomalía
        df.loc[start_index:end_index, axis] *= 5
        
        # Añadir un offset durante la anomalía
        df.loc[start_index:end_index, axis] += np.random.uniform(2, 4)
    
    return df

# Parámetros
duration = 600  # segundos
sample_rate = 200  # Hz
anomaly_percentage = 80  # porcentaje de la señal que será anómala

normal_signal = generate_normal_signal(duration, sample_rate)
anomalous_signal = generate_anomalous_signal(duration, sample_rate, anomaly_percentage)

print("Normal Signal:")
print(normal_signal.head())
print(normal_signal.tail())
print("\nAnomalous Signal:")
print(anomalous_signal.head())
print(anomalous_signal.tail())

# Para guardar los datos en archivos CSV:
normal_signal.to_csv('normal_signal.csv', index=False)
anomalous_signal.to_csv('anomalous_signal.csv', index=False)
