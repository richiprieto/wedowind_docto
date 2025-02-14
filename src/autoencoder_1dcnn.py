import torch
import torch.nn as nn

class Autoencoder1DCNN(nn.Module):
    def __init__(self, input_size, window_size):
        super(Autoencoder1DCNN, self).__init__()
        self.window_size = window_size
        self.num_features = input_size // window_size  # Calcula características basado en tamaño de ventana
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            
            nn.ConvTranspose1d(32, self.num_features, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Capa final para aplanar
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Reshape para CNN: (batch, features, window_size)
        x = x.view(-1, self.num_features, self.window_size)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Reconstruir a dimensión original
        decoded = self.flatten(decoded)
        return decoded 