import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# Autoencoder MLP
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid(),  # You can change this depending on the range of your input data
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Generación de datos simulados (dummy) - Reemplaza esto con tu dataset real
def generate_synthetic_data(num_samples, input_size):
    # Datos normales
    normal_data = np.random.normal(
        loc=0.5, scale=0.1, size=(num_samples // 2, input_size)
    )
    # Datos anómalos (con perturbaciones)
    anomaly_data = np.random.normal(
        loc=0.8, scale=0.1, size=(num_samples // 2, input_size)
    )

    # Concatenamos ambos
    data = np.vstack([normal_data, anomaly_data])

    return data


# Preparación de los datos
def prepare_dataloaders(data, batch_size=32):
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(
        data_tensor, data_tensor
    )  # Input and target are the same for autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Entrenamiento del autoencoder
def train_autoencoder(
    autoencoder, dataloader, epochs=100, learning_rate=0.001, device="cpu"
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    autoencoder.to(device)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return loss_history


# Visualización de las reconstrucciones
def visualize_reconstructions(autoencoder, dataloader, device="cpu"):
    autoencoder.eval()  # Set model to evaluation mode

    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = autoencoder(inputs)

        # Visualizar el primer batch de entradas y sus reconstrucciones
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        for i in range(5):  # Mostrar 5 ejemplos
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.plot(inputs[i])

            plt.subplot(1, 2, 2)
            plt.title("Reconstrucción")
            plt.plot(outputs[i])
            plt.show()

        break  # Solo mostrar un batch


# Detección de anomalías por reconstrucción
def detect_anomalies(autoencoder, data, threshold=0.1, device="cpu"):
    autoencoder.eval()  # Set model to evaluation mode

    data_tensor = torch.FloatTensor(data).to(device)
    with torch.no_grad():
        reconstructed = autoencoder(data_tensor)

    # Cálculo del error de reconstrucción
    mse_loss = nn.MSELoss(reduction="none")
    reconstruction_error = (
        mse_loss(reconstructed, data_tensor).mean(dim=1).cpu().numpy()
    )

    # Detección de anomalías: si el error es mayor que el umbral, es una anomalía
    anomalies = reconstruction_error > threshold
    return reconstruction_error, anomalies


# Main
if __name__ == "__main__":
    # Parámetros
    input_size = 10  # Ajusta esto de acuerdo a tu dataset
    num_samples = 1000
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    anomaly_threshold = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generación y preparación de datos
    data = generate_synthetic_data(num_samples, input_size)
    dataloader = prepare_dataloaders(data, batch_size=batch_size)

    # Inicializar el autoencoder
    autoencoder = Autoencoder(input_size)

    # Entrenar el autoencoder
    loss_history = train_autoencoder(
        autoencoder,
        dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    # Visualización de la curva de pérdida
    plt.plot(loss_history)
    plt.title("Curva de pérdida")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.show()

    # Visualizar las reconstrucciones
    visualize_reconstructions(autoencoder, dataloader, device=device)

    # Detección de anomalías
    reconstruction_error, anomalies = detect_anomalies(
        autoencoder, data, threshold=anomaly_threshold, device=device
    )

    # Mostrar resultados de las anomalías
    print("Errores de reconstrucción:", reconstruction_error)
    print("Anomalías detectadas:", anomalies)
