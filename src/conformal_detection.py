# conformal_detection.py
import torch
import numpy as np
from train_autoencoder import train_autoencoder
import pandas as pd


import torch


def conformal_anomaly_detection(
    autoencoder, data, significance_level=0.1, device="cpu"
):
    """
    Realiza la detección de anomalías utilizando Conformal Anomaly Detection (CAD).

    :param autoencoder: Modelo de autoencoder entrenado.
    :param data: Datos de entrada (DataFrame o numpy array).
    :param significance_level: Nivel de significancia para el cálculo del umbral.
    :param device: Dispositivo (CPU o GPU).
    :return: Errores de reconstrucción y umbral de conformidad.
    """
    # Asegurarse de que los datos estén en formato numpy array
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    data_tensor = torch.FloatTensor(data).to(
        device
    )  # Convertir los datos a tensor y mover a dispositivo

    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(data_tensor)

    mse_loss = torch.nn.MSELoss(reduction="none")
    reconstruction_error = (
        mse_loss(reconstructed, data_tensor).mean(dim=1).cpu().numpy()
    )

    # Obtener el percentil del error de reconstrucción como el umbral de conformidad
    q_hat = np.quantile(reconstruction_error, 1 - significance_level)

    return reconstruction_error, q_hat


def detect_conformal_anomalies(
    autoencoder,
    data,
    significance_level_prev=0.1,
    significance_level_total=0.05,
    device="cpu",
):
    """
    Detecta anomalías utilizando Conformal Anomaly Detection para dos niveles de fallos.

    :param autoencoder: Modelo de autoencoder entrenado.
    :param data: Datos de entrada (DataFrame o numpy array).
    :param significance_level_prev: Nivel de significancia para detectar un fallo previo.
    :param significance_level_total: Nivel de significancia para detectar un fallo total.
    :param device: Dispositivo (CPU o GPU).
    :return: Errores de reconstrucción, etiquetas de estado (0 = normal, 1 = fallo previo, 2 = fallo total),
             umbral para fallo previo y umbral para fallo total.
    """
    reconstruction_error, q_hat_prev = conformal_anomaly_detection(
        autoencoder, data, significance_level=significance_level_prev, device=device
    )

    _, q_hat_total = conformal_anomaly_detection(
        autoencoder, data, significance_level=significance_level_total, device=device
    )

    # Clasificar el estado de los datos basados en los umbrales
    state_labels = np.zeros_like(reconstruction_error)
    state_labels[reconstruction_error > q_hat_prev] = 1  # Fallo Previo
    state_labels[reconstruction_error > q_hat_total] = 2  # Fallo Total

    return reconstruction_error, state_labels, q_hat_prev, q_hat_total


if __name__ == "__main__":
    # Ejemplo de uso
    from read_hdf5 import load_hdf5_file
    from autoencoder_mlp import AutoencoderMLP

    file_path = "data/your_dataset.h5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)

    input_size = data.shape[1]
    model = AutoencoderMLP(input_size)

    # Supón que ya has entrenado el autoencoder
    model_type = "lstm" if isinstance(model, AutoencoderLSTM) else "mpl"

    significance_level_prev = 0.1
    significance_level_total = 0.05

    reconstruction_error, state_labels, q_hat_prev, q_hat_total = (
        detect_conformal_anomalies(
            model,
            data,
            significance_level_prev=significance_level_prev,
            significance_level_total=significance_level_total,
        )
    )

    # Guardar algunos resultados con el sufijo del modelo
    for i in range(10):
        print(
            f"Muestra {i}: Error de reconstrucción = {reconstruction_error[i]:.4f}, Estado = {state_labels[i]}, Modelo = {model_type}"
        )
