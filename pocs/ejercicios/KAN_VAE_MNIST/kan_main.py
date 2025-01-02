# Importar módulos necesarios
import sys

sys.path.append("./efficient-kan-master")
from src.efficient_kan import KAN

import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

# Establecer semilla para reproducibilidad
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Transformaciones para las imágenes
im_tfs = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5,), (0.5,))])

# Cargar el conjunto de entrenamiento y excluir los ceros (anomalías)
train_set = torchvision.datasets.MNIST(
    root="./mnist", train=True, download=True, transform=im_tfs
)
train_indices = train_set.targets != 1  # Filtrar dígitos que no son '0'
train_set.targets = train_set.targets[train_indices]
train_set.data = train_set.data[train_indices]

# Dividir el conjunto de entrenamiento en entrenamiento y validación
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_dataset, val_dataset = random_split(
    train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# Definir el autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = KAN([28 * 28, 256, 3])
        self.decoder = KAN([3, 256, 28 * 28])

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


net = Autoencoder()

# Función de pérdida y optimizador
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


def to_img(x):
    """
    Función para convertir la salida en imágenes.
    """
    x = 0.5 * (x + 1.0)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


# Implementación de early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.min_delta = min_delta
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Resetear contador si hay mejora
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Entrenamiento del autoencoder con early stopping
start = time.time()

num_epochs = 50
early_stopping = EarlyStopping(patience=5, min_delta=0.01)

for epoch in range(num_epochs):
    net.train()
    train_loss = 0
    for im, _ in train_loader:
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        # Propagación hacia adelante
        _, output = net(im)
        loss = criterion(output, im) / im.shape[0]  # Pérdida promedio por imagen
        # Propagación hacia atrás y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validación
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for im, _ in val_loader:
            im = im.view(im.shape[0], -1)
            im = Variable(im)
            _, output = net(im)
            loss = criterion(output, im) / im.shape[0]
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(
        "Epoch [{}/{}], Pérdida de Entrenamiento: {:.4f}, Pérdida de Validación: {:.4f}".format(
            epoch + 1, num_epochs, train_loss, val_loss
        )
    )

    # Comprobar early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Detención temprana en la época {}".format(epoch + 1))
        break

    # Guardar imágenes reconstruidas cada cierto número de épocas
    if (epoch + 1) % 5 == 0:
        pic = to_img(output.cpu().data)
        if not os.path.exists("./simple_autoencoder"):
            os.mkdir("./simple_autoencoder")
        save_image(pic, "./simple_autoencoder/image_{}.png".format(epoch + 1))

end = time.time()
print(f"Tiempo de entrenamiento: {end - start} segundos")

# Implementación de Conformal Prediction para detección de anomalías
# Paso 1: Calcular los errores de reconstrucción en el conjunto de validación
val_errors = []
net.eval()
with torch.no_grad():
    for im, _ in val_loader:
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        _, output = net(im)
        errors = torch.mean((output - im) ** 2, dim=1)
        val_errors.extend(errors.cpu().numpy())

# Paso 2: Establecer el nivel de significancia (alpha)
alpha = 0.05  # Podemos ajustar este valor según nuestras necesidades

# Paso 3: Calcular el umbral de conformidad
threshold = np.quantile(val_errors, 1 - alpha)
print(f"Umbral de detección (Conformal Prediction) con alpha={alpha}: {threshold}")

# Cargar el conjunto de prueba completo (incluyendo anomalías)
test_set = torchvision.datasets.MNIST(
    root="./mnist", train=False, download=True, transform=im_tfs
)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# Paso 4: Detectar anomalías en el conjunto de prueba utilizando el umbral
reconstruction_errors = []
labels = []
predictions = []
with torch.no_grad():
    for im, label in test_loader:
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        _, output = net(im)
        errors = torch.mean((output - im) ** 2, dim=1)
        reconstruction_errors.extend(errors.cpu().numpy())
        labels.extend(label.numpy())
        # Predicción: 1 si es anomalía, 0 si es normal
        preds = (errors > threshold).int()
        predictions.extend(preds.cpu().numpy())

# Analizar los resultados
results_df = pd.DataFrame(
    {
        "Error_de_reconstrucción": reconstruction_errors,
        "Etiqueta": labels,
        "Anomalía_predicha": predictions,
    }
)
results_df["Anomalía_real"] = (
    results_df["Etiqueta"] == 0
)  # Las anomalías son los dígitos '0'

# Evaluación del modelo
from sklearn.metrics import classification_report, confusion_matrix

print("Reporte de clasificación:")
print(
    classification_report(results_df["Anomalía_real"], results_df["Anomalía_predicha"])
)

print("Matriz de confusión:")
print(confusion_matrix(results_df["Anomalía_real"], results_df["Anomalía_predicha"]))

# Histograma de errores de reconstrucción
plt.figure(figsize=(10, 6))
plt.hist(
    results_df[results_df["Anomalía_real"] == False]["Error_de_reconstrucción"],
    bins=50,
    alpha=0.5,
    label="Normales",
)
plt.hist(
    results_df[results_df["Anomalía_real"] == True]["Error_de_reconstrucción"],
    bins=50,
    alpha=0.5,
    label="Anomalías (Ceros)",
)
plt.axvline(
    x=threshold, color="r", linestyle="--", label="Umbral de Conformal Prediction"
)
plt.legend()
plt.xlabel("Error de Reconstrucción (MSE por imagen)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores de Reconstrucción")
plt.show()

# Visualización en 3D de las características codificadas
view_data = Variable(
    (test_set.data[:200].type(torch.FloatTensor).view(-1, 28 * 28) / 255.0 - 0.5) / 0.5
)
encode, _ = net(view_data)  # Extraer características comprimidas
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
values = test_set.targets[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, str(s), backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
