import torch
import torch.nn as nn
from kan import KAN, KANLinear  # Asegúrate de que el archivo kan.py esté en el mismo directorio o ajusta la ruta de importación según corresponda


class AutoencoderKAN(nn.Module):
    def __init__(self, input_size, grid_size=5, spline_order=3):
        super(AutoencoderKAN, self).__init__()
        
        # Definir las capas del encoder usando KANLinear
        self.encoder = nn.Sequential(
            KANLinear(
                in_features=input_size,
                out_features=32,
                grid_size=grid_size,
                spline_order=spline_order
            ),
            nn.ReLU(),
            KANLinear(
                in_features=32,
                out_features=16,
                grid_size=grid_size,
                spline_order=spline_order
            ),
            nn.ReLU(),
            KANLinear(
                in_features=16,
                out_features=8,
                grid_size=grid_size,
                spline_order=spline_order
            )
        )
        
        # Definir las capas del decoder usando KANLinear
        self.decoder = nn.Sequential(
            KANLinear(
                in_features=8,
                out_features=16,
                grid_size=grid_size,
                spline_order=spline_order
            ),
            nn.ReLU(),
            KANLinear(
                in_features=16,
                out_features=32,
                grid_size=grid_size,
                spline_order=spline_order
            ),
            nn.ReLU(),
            KANLinear(
                in_features=32,
                out_features=input_size,
                grid_size=grid_size,
                spline_order=spline_order
            )
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Ejemplo de uso
    input_size = 10  # Cambia este valor según tu dataset
    model = AutoencoderKAN(input_size=input_size)
    print(model)
