# autoencoder_lstm.py
import torch
import torch.nn as nn

class AutoencoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AutoencoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Codificación
        _, (hidden, cell) = self.encoder(x)

        # Decodificación
        decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_size).to(x.device)
        output, _ = self.decoder(decoder_input, (hidden, cell))

        # Proyección al espacio original
        output = self.output_layer(output)

        return output

if __name__ == "__main__":
    # Ejemplo de uso
    input_size = 10
    hidden_size = 64
    num_layers = 2
    seq_length = 20
    batch_size = 32
    
    model = AutoencoderLSTM(input_size, hidden_size, num_layers)
    print(model)
    
    # Ejemplo de entrada
    sample_input = torch.randn(batch_size, seq_length, input_size)
    output = model(sample_input)
    print(f"Forma de la entrada: {sample_input.shape}")
    print(f"Forma de la salida: {output.shape}")
