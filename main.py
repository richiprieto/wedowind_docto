# main.py
from read_hdf5 import load_hdf5_file
from autoencoder_mlp import AutoencoderMLP
from train_autoencoder import train_autoencoder
from conformal_detection import detect_conformal_anomalies


def main():
    # Cargar datos
    file_path = "../dataset/Aventa_Taggenberg_01_11_2022.hdf5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)

    # Definir el tamaño de la entrada
    input_size = data.shape[1]

    # Inicializar y entrenar el modelo
    model = AutoencoderMLP(input_size)
    train_autoencoder(model, data, epochs=50, learning_rate=0.001, batch_size=32)

    # Detectar anomalías con conformal anomaly detection
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

    # Mostrar resultados
    for i in range(10):
        estado = (
            "Normal"
            if state_labels[i] == 0
            else ("Fallo Previo" if state_labels[i] == 1 else "Fallo Total")
        )
        print(
            f"Muestra {i}: Error de reconstrucción = {reconstruction_error[i]:.4f}, Estado = {estado}"
        )


if __name__ == "__main__":
    main()
