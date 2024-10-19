# autoencoder_mlp.py
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class AutoencoderMLP(nn.Module):
    def __init__(self, input_size):
        super(AutoencoderMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x):
        x = x.to(
            next(self.parameters()).device
        )  # Mover los datos al dispositivo del modelo
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Ejemplo de uso
    input_size = 10  # Cambia este valor según tu dataset
    model = AutoencoderMLP(input_size)
    print(model)
