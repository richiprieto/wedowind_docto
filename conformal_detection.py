# conformal_detection.py
import torch
import numpy as np
from train_autoencoder import train_autoencoder


def conformal_anomaly_detection(
    autoencoder, data, significance_level=0.1, device="cpu"
):
    autoencoder.eval()

    # Calcular los errores de reconstrucción para las muestras del dataset normal
    data_tensor = torch.FloatTensor(data).to(device)
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
    reconstruction_error, q_hat_prev = conformal_anomaly_detection(
        autoencoder, data, significance_level=significance_level_prev, device=device
    )

    _, q_hat_total = conformal_anomaly_detection(
        autoencoder, data, significance_level=significance_level_total, device=device
    )

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

    # Mostrar algunos resultados
    for i in range(10):
        print(
            f"Muestra {i}: Error de reconstrucción = {reconstruction_error[i]:.4f}, Estado = {state_labels[i]}"
        )
