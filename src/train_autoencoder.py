# train_autoencoder.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from autoencoder_lstm import AutoencoderLSTM
import pandas as pd


def train_autoencoder(
    model, data, epochs=50, learning_rate=0.001, batch_size=32, device="cpu"
):
    """
    Entrena el autoencoder con los datos proporcionados.

    :param model: Modelo del autoencoder.
    :param data: Datos de entrenamiento (numpy array o tensor) de forma (n_seqs, seq_length, input_size).
    :param epochs: Número de épocas de entrenamiento.
    :param learning_rate: Tasa de aprendizaje.
    :param batch_size: Tamaño del batch.
    :param device: Dispositivo para entrenar (CPU o GPU).
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    data_tensor = torch.FloatTensor(data).to(device)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    loss_history = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, loss_history


if __name__ == "__main__":
    # Ejemplo de uso
    from read_hdf5 import load_hdf5_file

    file_path = "data/your_dataset.h5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)

    input_size = data.shape[1]
    model = AutoencoderLSTM(input_size)
    train_autoencoder(model, data, epochs=50, learning_rate=0.001, batch_size=32)
