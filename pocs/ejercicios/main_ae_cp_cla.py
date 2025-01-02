import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union

# Configuración
SEED = 42
BATCH_SIZE = 128
EMBEDDING_DIM = 64
NUM_EPOCHS = 100
RESULTS_DIR = "results"

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = EMBEDDING_DIM):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(decoder_input)
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = NUM_EPOCHS) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    early_stopping = EarlyStopping(patience=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            X_pred = model(X_batch)
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch in val_loader:
                X_batch = X_batch.to(device)
                X_pred = model(X_batch)
                loss = criterion(X_pred, X_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history

def get_reconstruction_errors(model: nn.Module, loader: DataLoader) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            X_pred = model(X_batch)
            loss = nn.functional.mse_loss(X_pred, X_batch, reduction='none').mean(dim=(1,2))
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

class ConformalAnomalyDetector:
    def __init__(self, calibration_errors: np.ndarray, significance: float = 0.05):
        self.calibration_errors = calibration_errors
        self.significance = significance
        self.threshold = np.percentile(calibration_errors, (1 - significance) * 100)

    def detect_anomalies(self, test_errors: np.ndarray) -> np.ndarray:
        return test_errors > self.threshold

    def set_significance(self, significance: float):
        self.significance = significance
        self.threshold = np.percentile(self.calibration_errors, (1 - significance) * 100)

    def evaluate(self, test_errors: np.ndarray, true_anomalies: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        predicted_anomalies = self.detect_anomalies(test_errors)
        
        accuracy = accuracy_score(true_anomalies, predicted_anomalies)
        precision = precision_score(true_anomalies, predicted_anomalies, zero_division=0)
        recall = recall_score(true_anomalies, predicted_anomalies, zero_division=0)
        f1 = f1_score(true_anomalies, predicted_anomalies, zero_division=0)
        
        # Verificar si hay más de una clase antes de calcular AUC ROC
        if len(np.unique(true_anomalies)) > 1:
            auc_roc = roc_auc_score(true_anomalies, predicted_anomalies)
            fpr, tpr, _ = roc_curve(true_anomalies, predicted_anomalies)
        else:
            auc_roc = None
            fpr, tpr = None, None
        
        cm = confusion_matrix(true_anomalies, predicted_anomalies)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        }

def plot_results(train_errors: np.ndarray, val_errors: np.ndarray, test_errors: np.ndarray, 
                 threshold: float, predictions: np.ndarray, significance: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Distribución de errores
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='Train')
    plt.hist(val_errors, bins=50, alpha=0.5, label='Validation')
    plt.hist(test_errors, bins=50, alpha=0.5, label='Test')
    plt.axvline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'error_distribution_sig_{significance}.png'))
    plt.close()
    
    # Curva ROC
    try:
        fpr, tpr, _ = roc_curve(np.concatenate([np.zeros_like(train_errors), np.ones_like(test_errors)]),
                                np.concatenate([train_errors, test_errors]))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Significance = {significance})')
        plt.legend(loc="lower right")
    except ValueError:
        print(f"No se pudo calcular la curva ROC para significancia {significance}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No se pudo calcular la curva ROC\nTodos los valores son de la misma clase", 
                 ha='center', va='center', fontsize=12)
        plt.title(f'Curva ROC no disponible (Significancia = {significance})')

    plt.savefig(os.path.join(RESULTS_DIR, f'roc_curve_sig_{significance}.png'))
    plt.close()
    
    # Serie temporal de errores
    plt.figure(figsize=(12, 6))
    plt.plot(test_errors, label='Reconstruction Error')
    plt.axhline(threshold, color='r', linestyle='--', label='CP Threshold')
    plt.xlabel('Sample')
    plt.ylabel('Reconstruction Error')
    plt.title(f'Time Series of Reconstruction Errors (Significance = {significance})')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'error_time_series_sig_{significance}.png'))
    plt.close()
    
    # Matriz de confusión
    cm = confusion_matrix(np.concatenate([np.zeros_like(train_errors), np.ones_like(test_errors)]), 
                          np.concatenate([train_errors > threshold, predictions]))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Significance = {significance})')
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_sig_{significance}.png'))
    plt.close()

def main():
    set_seed()
    
    # Cargar y procesar datos
    df_train = pd.read_csv('normal_signal.csv')
    df_test = pd.read_csv('anomalous_signal.csv')
    
    # Preparar datos para el autoencoder
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.drop('Time', axis=1))
    X_test = scaler.transform(df_test.drop('Time', axis=1))
    
    # Configurar parámetros
    seq_len = 10  # Longitud de la secuencia para el LSTM
    n_features = X_train.shape[1]
    
    # Crear datasets y dataloaders
    train_dataset = TimeSeriesDataset(X_train, seq_len)
    test_dataset = TimeSeriesDataset(X_test, seq_len)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Configurar y entrenar el modelo
    model = LSTMAutoencoder(seq_len, n_features, EMBEDDING_DIM)
    history = train_model(model, train_loader, val_loader)
    
    # Evaluar el modelo y aplicar Conformal Prediction
    train_errors = get_reconstruction_errors(model, train_loader)
    val_errors = get_reconstruction_errors(model, val_loader)
    test_errors = get_reconstruction_errors(model, test_loader)
    
    calibration_errors = np.concatenate([train_errors, val_errors])
    
    significances = [0.01, 0.05, 0.1, 0.2]
    all_metrics = []
    
    for significance in significances:
        conformal_detector = ConformalAnomalyDetector(calibration_errors, significance)
        predictions = conformal_detector.detect_anomalies(test_errors)
        
        # Suponemos que todas las muestras en el conjunto de prueba son anómalas
        true_anomalies = np.ones_like(test_errors, dtype=bool)
        
        metrics = conformal_detector.evaluate(test_errors, true_anomalies)
        all_metrics.append(metrics)
        
        print(f"\nMétricas para nivel de significancia {significance}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Generar y guardar gráficos
        plot_results(train_errors, val_errors, test_errors, conformal_detector.threshold, predictions, significance)
    
    # Generar gráfico de resumen de métricas
    plot_metrics_summary(all_metrics, significances)
    
    print(f"\nResultados y gráficos guardados en el directorio '{RESULTS_DIR}'.")

def plot_metrics_summary(metrics_list, significances):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    data = {metric: [m[metric] if m[metric] is not None else 0 for m in metrics_list] for metric in metrics}
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(significances))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, data[metric], width, label=metric)
    
    plt.xlabel('Nivel de Significancia')
    plt.ylabel('Valor de la Métrica')
    plt.title('Resumen de Métricas para Diferentes Niveles de Significancia')
    plt.xticks(x + width*2, significances)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_summary.png'))
    plt.close()

def plot_loss_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_history.png'))
    plt.close()

def save_results(results: Dict[str, Union[float, np.ndarray]], file_name: str):
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, file_name), index=False)
    print(f"Resultados guardados en {os.path.join(RESULTS_DIR, file_name)}")

def main():
    set_seed()
    
    # Cargar y procesar datos
    df_train = pd.read_csv('normal_signal.csv')
    df_test = pd.read_csv('anomalous_signal.csv')
    
    # Preparar datos para el autoencoder
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.drop('Time', axis=1))
    X_test = scaler.transform(df_test.drop('Time', axis=1))
    
    # Configurar parámetros
    seq_len = 10  # Longitud de la secuencia para el LSTM
    n_features = X_train.shape[1]
    
    # Crear datasets y dataloaders
    train_dataset = TimeSeriesDataset(X_train, seq_len)
    test_dataset = TimeSeriesDataset(X_test, seq_len)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Configurar y entrenar el modelo
    model = LSTMAutoencoder(seq_len, n_features, EMBEDDING_DIM)
    history = train_model(model, train_loader, val_loader)
    
    # Graficar historial de pérdida
    plot_loss_history(history)
    
    # Evaluar el modelo y aplicar Conformal Prediction
    train_errors = get_reconstruction_errors(model, train_loader)
    val_errors = get_reconstruction_errors(model, val_loader)
    test_errors = get_reconstruction_errors(model, test_loader)
    
    calibration_errors = np.concatenate([train_errors, val_errors])
    
    significances = [0.025, 0.05, 0.25, 0.5]
    all_metrics = []
    all_results = []
    
    for significance in significances:
        conformal_detector = ConformalAnomalyDetector(calibration_errors, significance)
        predictions = conformal_detector.detect_anomalies(test_errors)
        
        # Suponemos que todas las muestras en el conjunto de prueba son anómalas
        true_anomalies = np.ones_like(test_errors, dtype=bool)
        
        metrics = conformal_detector.evaluate(test_errors, true_anomalies)
        all_metrics.append(metrics)
        
        print(f"\nMétricas para nivel de significancia {significance}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Generar y guardar gráficos
        plot_results(train_errors, val_errors, test_errors, conformal_detector.threshold, predictions, significance)
        
        # Guardar resultados
        results = {
            "reconstruction_error": test_errors,
            "is_anomaly": predictions.astype(int),
            "threshold": conformal_detector.threshold,
            "significance": significance,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        all_results.append(results)
    
    # Generar gráfico de resumen de métricas
    plot_metrics_summary(all_metrics, significances)
    
    # Guardar todos los resultados
    combined_results = pd.concat([pd.DataFrame(r) for r in all_results])
    save_results(combined_results, "anomaly_detection_results.csv")
    
    print(f"\nResultados y gráficos guardados en el directorio '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()