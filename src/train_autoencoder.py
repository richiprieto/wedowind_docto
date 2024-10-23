# train_autoencoder.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from autoencoder_lstm import AutoencoderLSTM
import pandas as pd


def train_autoencoder(
    model, train_data, val_data, epochs=50, learning_rate=0.001, batch_size=32, patience=5, device="cpu"
):
    """
    Entrena el autoencoder y aplica early stopping basado en la pérdida de validación.

    :param model: Modelo de autoencoder.
    :param train_data: Datos de entrenamiento (numpy array).
    :param val_data: Datos de validación (numpy array).
    :param epochs: Número de épocas de entrenamiento.
    :param learning_rate: Tasa de aprendizaje.
    :param batch_size: Tamaño de batch.
    :param patience: Número de épocas para el early stopping.
    :param device: Dispositivo (CPU o GPU).
    :return: Modelo entrenado y el historial de pérdidas.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(
        torch.FloatTensor(train_data),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torch.FloatTensor(val_data),
        batch_size=batch_size,
        shuffle=False
    )

    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_data.size(0)

        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                val_loss += loss.item() * batch_data.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Época [{epoch+1}/{epochs}], Pérdida de entrenamiento: {epoch_loss:.6f}, Pérdida de validación: {val_loss:.6f}")

        # Comprobación para early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping activado.")
                break

    # Restaurar el mejor modelo
    model.load_state_dict(best_model_state)

    # Guardar el modelo
    torch.save(model.state_dict(), 'output/best_model.pth')

    return model, loss_history, val_loss_history


if __name__ == "__main__":
    # Ejemplo de uso
    from read_hdf5 import load_hdf5_file

    file_path = "data/your_dataset.h5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)

    input_size = data.shape[1]
    model = AutoencoderLSTM(input_size)
    train_autoencoder(model, data, epochs=50, learning_rate=0.001, batch_size=32)
