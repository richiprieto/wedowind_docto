# autoencoder_mlp.py
import torch
import torch.nn as nn


class AutoencoderMLP(nn.Module):
    def __init__(self, input_size):
        super(AutoencoderMLP, self).__init__()

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
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Ejemplo de uso
    input_size = 10  # Cambia este valor según tu dataset
    model = AutoencoderMLP(input_size)
    print(model)
