import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesGenerator:
    """
    Generates synthetic time series data with optional anomalies.
    """
    def __init__(self, n_samples=1000, freq=1/50, amplitude=1, noise_level=0.1):
        self.n_samples = n_samples
        self.freq = freq
        self.amplitude = amplitude
        self.noise_level = noise_level

    def generate_normal_data(self):
        t = np.linspace(0, self.n_samples, self.n_samples)
        y = self.amplitude * np.sin(2 * np.pi * self.freq * t)
        y += np.random.normal(0, self.noise_level, self.n_samples)
        return y

    def add_anomalies(self, data, n_anomalies=50, anomaly_std=3):
        anomaly_indices = np.random.choice(range(len(data)), n_anomalies, replace=False)
        data_with_anomalies = data.copy()
        data_with_anomalies[anomaly_indices] += np.random.normal(0, anomaly_std, n_anomalies)
        return data_with_anomalies, anomaly_indices

class TimeSeriesICM:
    """
    Inductive Conformal Measure (ICM) for time series data using ARIMA predictions.
    """
    def __init__(self, order=(1,1,1)):
        self.order = order

    def fit(self, data):
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()

    def __call__(self, data):
        predictions = self.fitted_model.forecast(steps=len(data))
        return np.abs(data - predictions)

class ConformalAnomalyDetector:
    """
    Conformal Anomaly Detector for Time Series data.
    """
    def __init__(self, ICM, calibration_data, significance=0.05):
        self.ICM = ICM
        self.calibration_data = calibration_data
        self.significance = significance
        self.ICM.fit(calibration_data)
        self.calibration_scores = self.ICM(calibration_data)

    def detect_anomalies(self, test_data):
        test_scores = self.ICM(test_data)
        threshold = np.percentile(self.calibration_scores, (1 - self.significance) * 100)
        return test_scores > threshold

    def set_significance(self, significance):
        self.significance = significance

    def evaluate(self, test_data, true_anomalies):
        predicted_anomalies = self.detect_anomalies(test_data)
        
        accuracy = accuracy_score(true_anomalies, predicted_anomalies)
        precision = precision_score(true_anomalies, predicted_anomalies)
        recall = recall_score(true_anomalies, predicted_anomalies)
        f1 = f1_score(true_anomalies, predicted_anomalies)
        auc_roc = roc_auc_score(true_anomalies, predicted_anomalies)
        
        fpr, tpr, _ = roc_curve(true_anomalies, predicted_anomalies)
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

# Plotting functions remain largely the same, just update titles and filenames
def plot_roc_curve(fpr, tpr, auc_roc, significance):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Significance Level = {significance})')
    plt.legend(loc="lower right")
    plt.savefig(f'results/ts_roc_curve_sig_{significance}.png')
    plt.close()

def plot_confusion_matrix(cm, significance):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Significance Level = {significance})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/ts_confusion_matrix_sig_{significance}.png')
    plt.close()

def plot_metrics_summary(metrics_list, significances):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    data = {metric: [m[metric] for m in metrics_list] for metric in metrics}
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(significances))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, data[metric], width, label=metric)
    
    plt.xlabel('Significance Level')
    plt.ylabel('Metric Value')
    plt.title('Metrics Summary for Different Significance Levels')
    plt.xticks(x + width*2, significances)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=5)
    plt.tight_layout()
    plt.savefig('results/ts_metrics_summary.png')
    plt.close()

def plot_time_series(data, anomalies, predicted_anomalies, significance):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Time Series')
    plt.scatter(np.where(anomalies)[0], data[anomalies], color='red', label='True Anomalies')
    plt.scatter(np.where(predicted_anomalies)[0], data[predicted_anomalies], color='green', marker='x', label='Predicted Anomalies')
    plt.title(f'Time Series with Anomalies (Significance Level = {significance})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'results/ts_anomalies_sig_{significance}.png')
    plt.close()

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    np.random.seed(42)
    
    # Generate time series data
    ts_generator = TimeSeriesGenerator(n_samples=1000)
    normal_data = ts_generator.generate_normal_data()
    data_with_anomalies, anomaly_indices = ts_generator.add_anomalies(normal_data)
    
    # Create true anomaly labels
    true_anomalies = np.zeros(len(data_with_anomalies), dtype=bool)
    true_anomalies[anomaly_indices] = True
    
    # Split data
    train_data, test_data, train_anomalies, test_anomalies = train_test_split(
        data_with_anomalies, true_anomalies, test_size=0.3, random_state=42
    )
    
    # Initialize ICM and Conformal Anomaly Detector
    icm = TimeSeriesICM(order=(1,1,1))
    conformal_detector = ConformalAnomalyDetector(icm, train_data)
    
    significances = [0.025, 0.05, 0.25, 0.5] 
    all_metrics = []
    
    for significance in significances:
        conformal_detector.set_significance(significance)
        
        # Evaluate the model
        metrics = conformal_detector.evaluate(test_data, test_anomalies)
        all_metrics.append(metrics)
        
        print(f"\nMetrics for significance level {significance}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Generate and save plots
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc_roc'], significance)
        plot_confusion_matrix(metrics['confusion_matrix'], significance)
        
        # Plot time series with anomalies
        predicted_anomalies = conformal_detector.detect_anomalies(test_data)
        plot_time_series(test_data, test_anomalies, predicted_anomalies, significance)

    # Generate summary metrics plot
    plot_metrics_summary(all_metrics, significances)

if __name__ == '__main__':
    main()