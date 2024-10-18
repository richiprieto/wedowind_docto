# train_autoencoder.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from autoencoder_mlp import AutoencoderMLP


def train_autoencoder(
    model, data, epochs=100, learning_rate=0.001, batch_size=32, device="cpu"
):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(
        data_tensor, data_tensor
    )  # Input y target son lo mismo en autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data in dataloader:
            inputs, _ = batch_data
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Ejemplo de uso
    from read_hdf5 import load_hdf5_file

    file_path = "data/your_dataset.h5"
    dataset_name = "SCADA_data"
    data = load_hdf5_file(file_path, dataset_name)

    input_size = data.shape[1]
    model = AutoencoderMLP(input_size)
    train_autoencoder(model, data, epochs=50, learning_rate=0.001, batch_size=32)
