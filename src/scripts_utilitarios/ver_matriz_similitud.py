import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_matriz_similitud(archivo_excel):
    """
    Visualiza la matriz de similitud desde un archivo Excel en un gráfico de matriz.
    """
    # Leer la matriz de similitud desde el archivo Excel
    df = pd.read_excel(archivo_excel, index_col=0)

    # Crear el gráfico de matriz de similitud
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="viridis", cbar=True)
    plt.title("Matriz de Similitud")
    plt.xlabel("Archivos HDF5")
    plt.ylabel("Archivos HDF5")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    archivo_excel = "matriz_similitud.xlsx"
    visualizar_matriz_similitud(archivo_excel)
