import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from scipy.stats import norm

# 1. Cargar los datos
data = pd.read_excel("input/AirQualityUCI.xlsx")

# 2. Convertir 'Date' y 'Time' a cadenas de texto y combinarlas
data['Date'] = data['Date'].astype(str)
data['Time'] = data['Time'].astype(str)

# Intentar parsear sin especificar el formato para identificar problemas
data['Date_Time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')

# Verificar que la conversión fue exitosa
if data['Date_Time'].isnull().any():
    print("Advertencia: Algunas fechas no se pudieron parsear y fueron convertidas a NaT")

# Eliminar filas con NaT en 'Date_Time'
data.dropna(subset=['Date_Time'], inplace=True)

# Establecer 'Date_Time' como índice
data.set_index('Date_Time', inplace=True)

# 3. Limpiar datos
# Reemplazar valores -200 con NaN
data.replace(-200, np.nan, inplace=True)

# Eliminar columnas innecesarias
data.drop(columns=[
    'Date', 'Time', 'NMHC(GT)', 'C6H6(GT)', 'THC(GT)', 'PT08.S1(CO)', 
    'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
    'T', 'RH', 'AH'
], inplace=True, errors='ignore')

# 4. Resampling a nivel horario y manejo de valores faltantes
data_resampled = data.resample('h').mean()

# Imputar valores faltantes usando la media de cada columna
imputer = SimpleImputer(strategy='mean')
data_resampled_imputed = pd.DataFrame(imputer.fit_transform(data_resampled),
                                     columns=data_resampled.columns,
                                     index=data_resampled.index)

# Escalar los datos
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_resampled_imputed),
                           columns=data_resampled_imputed.columns,
                           index=data_resampled_imputed.index)

# 5. Detección de anomalías usando Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
data_scaled['anomaly_iforest'] = model.fit_predict(data_scaled)

# Identificar anomalías
anomalies_iforest = data_scaled[data_scaled['anomaly_iforest'] == -1]

# 6. Implementación de Conformal Anomaly Detection (CAD)

# Función para CAD
def conformal_anomaly_detection(X_train, X_test, alpha=0.05):
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    
    # Estimación del intervalo de confianza
    z_alpha = norm.ppf(1 - alpha / 2)
    lower_bound = mean_train - z_alpha * std_train
    upper_bound = mean_train + z_alpha * std_train
    
    # Detección de anomalías
    anomalies = ((X_test < lower_bound) | (X_test > upper_bound)).any(axis=1).astype(int) * -1 + 1
    
    return anomalies

# Dividir los datos en entrenamiento y test para aplicar CAD
train_size = int(len(data_scaled) * 0.8)
X_train = data_scaled.iloc[:train_size].drop(columns=['anomaly_iforest'])
X_test = data_scaled.iloc[train_size:].drop(columns=['anomaly_iforest'])

# Aplicar CAD
alpha = 0.05  # Nivel de confianza del 95%
anomalies_cad = conformal_anomaly_detection(X_train, X_test, alpha=alpha)

# Añadir resultados de CAD a los datos
data_scaled['anomaly_cad'] = 1
data_scaled['anomaly_cad'].iloc[train_size:] = anomalies_cad.values

anomalies_cad_df = data_scaled[data_scaled['anomaly_cad'] == -1]

# 7. Evaluación de los modelos
# Supongamos que tienes etiquetas verdaderas en una columna llamada 'true_labels'
# 1 para normal, -1 para anomalía
# true_labels = ...

# Evaluación de Isolation Forest
# print("Classification Report for Isolation Forest:")
# print(classification_report(true_labels, data_scaled['anomaly_iforest']))

# Evaluación de Conformal Anomaly Detection
# print("Classification Report for Conformal Anomaly Detection:")
# print(classification_report(true_labels[train_size:], data_scaled['anomaly_cad'].iloc[train_size:]))

# 8. Visualización de las anomalías

plt.figure(figsize=(14, 7))
for column in data_scaled.columns:
    if column not in ['anomaly_iforest', 'anomaly_cad']:
        plt.plot(data_scaled.index, data_scaled[column], label=column)

plt.scatter(anomalies_iforest.index, data_scaled.loc[anomalies_iforest.index, data_scaled.columns[0]], 
            color='red', label='Anomalías Isolation Forest', marker='x')
plt.scatter(anomalies_cad_df.index, data_scaled.loc[anomalies_cad_df.index, data_scaled.columns[0]], 
            color='blue', label='Anomalías CAD', marker='o')
plt.legend()
plt.title('Detección de Anomalías utilizando Todas las Variables')
plt.xlabel('Fecha y Hora')
plt.ylabel('Valores Escalados')
plt.show()
