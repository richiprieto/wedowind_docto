# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

# Data Generator Class
class DataGenerator():
    """
        Fake data generator that generates data along an n-dimensional hypercube. By default, 2-dimensional hypercube is used. Within-class variation
        achieved by simply adding Gaussian noise.
        @params
            num_classes: int
                The number of classes to generate
            num_samples_per_class: int
                The number of samples to generate per class (total samples = num_classes*num_samples_per_class)
            num_features: int
                The number of features to generate. Note: This can only be visualized if num_features = 2
            minFeature: int
                The minimum of our hyper-cube along each dimension
            maxFeature: int
                The maximum of our hyper-cube along each dimension
    """
    def __init__(self,num_classes=10,num_samples_per_class=100,num_features = 2,minFeature=0,maxFeature=50):
        self._num_classes = num_classes
        self._num_features = num_features
        self._num_samples_per_class = num_samples_per_class
        self._minFeature = minFeature
        self._maxFeature = maxFeature
        self.x = np.zeros((self._num_classes*self._num_samples_per_class,self._num_features))
        self.y = np.zeros((self._num_classes*self._num_samples_per_class,1))
        self._list_of_funcs = []
        for i in range(self._num_classes):
            features = np.random.randint(self._minFeature,self._maxFeature,(1,self._num_features)).astype(np.float64)
            noise = np.random.rand(self._num_samples_per_class,self._num_features)*6 # add some noise to the features
            features = features + noise
            self._list_of_funcs.append(features)
            self.x[i*self._num_samples_per_class:(i+1)*self._num_samples_per_class] = features
            self.y[i*self._num_samples_per_class:(i+1)*self._num_samples_per_class] = int(i)

    """
        Create a scatter plot of mx2 data with label in y for legend creation and title
        @params
            x: np.ndarray
                A mx2 array with m samples where containing x axis and y axis scatter plot data
            y: np.ndarray
                A m array of labels
            title: str
                The title of the scatter plot
        @return
            The generated plot in case you want to plot over it
    """
    def create_scatter_plot(self,x,y,title):
        assert x.shape[1] == 2,\
            print('Cannot create a scatter plot of data with {} features'.format(x.shape[1]))
        fig, ax = plt.subplots()
        scatter = ax.scatter(x[:,0], x[:,1], c=np.squeeze(y),alpha=0.3)
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.title(title)
        return plt

    """
        Generates data randomly across n-dimensional hypercube
        @params
            num_anomalies: int
                The number of anomalies to create
        @return: np.ndarray
            A num_anomalies x num_features array of the anomalies
    """
    def create_anomaly(self,num_anomalies):
        anomalies = np.random.randint(self._minFeature,self._maxFeature,(num_anomalies,self._num_features))
        return anomalies

    """
        Displays the anomolous data over the fake data set. Anomalies are displayed by a red cross and non-anomalies displayed by blue cross
        @params
            anomalies: np.ndarray
                An mxnum_features array
            isAnomaly: np.ndarray
                An mx1 boolean array indicating if the i'th sample is an anomaly or not
            block: bool
                Whether or not the plot will block further code execution
            title: str
                The title of the plot
    """
    def showAnomalies(self,anomalies,isAnomaly,block=False,title=''):
        self.plt = self.create_scatter_plot(self.x,self.y,title=title)
        yes_anomaly = anomalies[isAnomaly]
        not_anomaly = anomalies[np.invert(isAnomaly)]
        self.plt.scatter(yes_anomaly[:,0],yes_anomaly[:,1],marker='x',c='r',s=plt.rcParams['lines.markersize']**2.5,label='anomaly')
        self.plt.scatter(not_anomaly[:,0],not_anomaly[:,1],marker='x',c='b',s=plt.rcParams['lines.markersize']**2.5,label='not_anomaly')
        self.plt.legend()
        self.plt.show(block=block)

# KNearestNeighborsClass
class KNearestNeighbors():
    """
        A simple real-valued function to compute the conformal scores
        Each conformal score is the average k-nearest neighbors according to a specified metric
        @params
            k: int
                Determines k nearest neighbors
            metric: str
                distance metric (see scipy's pdist function for valid metrics)
    """
    def __init__(self,k,metric='euclidean'):
        self._k = k
        self._metric = metric

    """
        Returns a pairwise distance matrix
        @params
            x: np.ndarray
                An m x n array with m samples and n dimensions
    """
    def get_pairwise_distance_matrix(self,x):
        distances = pdist(x,self._metric)
        distance_matrix = squareform(distances)
        return distance_matrix

    """
        Returns the mean pairwise distance between the k'th nearest neighbors
        @params
            x: np.ndarray
                An m x n array with m samples and n dimensions
    """
    def __call__(self,x):
        distance_matrix = self.get_pairwise_distance_matrix(x)
        distance_matrix = np.sort(distance_matrix,axis=1)
        assert self._k +1 < distance_matrix.shape[1],\
            print('K must be less than the number of data points (k={},num_samples={})'.format(self._k +1,distance_matrix.shape[1]))
        return np.mean(distance_matrix[:,1:self._k+1],axis=1)

class ConformalAnomalyDetector():
    """
    Conformal Anomaly Detector Class
    @params
        ICM: class
            An object whose call operation should produce an array of conformal scores
        z: tuple (len==2)
            Each element is an (x,y) pair of the training set for CAD
        significance: float
            The significance level (must be between 0 and 1 exclusive)
    """
    def __init__ (self,ICM,z,significance=0.05):
        self._ICM = ICM
        self.x = z[0]
        self.y = z[1]
        assert significance > 0 and significance < 1, \
            print('Significance must be in range (0,1).')
        self._significance = significance
        
    """
    Return true or false if the test example are an anomaly
    @params
        test: np.ndarray
            A 1xn test example where m is the number of test examples and n is the number of dimensions
    @return: bool
        True if test input is anomaly and false otherwise 
    """
    def testIfAnomaly(self,test):
        conformal_set = np.concatenate((self.x,test))
        conformal_scores = self._ICM(conformal_set)
        p = np.sum(conformal_scores >= conformal_scores[-1]) / (len(self.y)+1)
        return p < self._significance

    """
    Return array of true or false if the test examples are an anomaly
    @params
        test: np.ndarray
            A mxn test example where m is the number of test examples and n is the number of dimensions
    @return: np.ndarray
        An mx1 array of true if test input is anomaly and false otherwise 
    """ 
    def __call__(self,anomalies):
        isAnomaly = [self.testIfAnomaly(np.expand_dims(anomalies[i],axis=0)) for i in range(anomalies.shape[0])]
        #print(isAnomaly)
        return isAnomaly

    """
    Change significance level (hyper-parameter)
    @params
        significance: float
            The significance level (must be between 0 and 1 exclusive)
    """ 
    def set_significance(self,significance):
        assert significance > 0 and significance < 1, \
            print('Significance must be in range (0,1).')
        self._significance = significance

    def evaluate(self, X_test, y_true):
        """
        Evalúa el rendimiento del detector de anomalías
        @params
            X_test: np.ndarray
                Conjunto de datos de prueba
            y_true: np.ndarray
                Etiquetas verdaderas (1 para anomalía, 0 para normal)
        @return: dict
            Diccionario con varias métricas de rendimiento y datos para gráficos
        """
        y_pred = self.__call__(X_test)
        
        # Convertir booleanos a enteros (True -> 1, False -> 0)
        y_pred_int = np.array(y_pred).astype(int)
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred_int)
        precision = precision_score(y_true, y_pred_int)
        recall = recall_score(y_true, y_pred_int)
        f1 = f1_score(y_true, y_pred_int)
        auc_roc = roc_auc_score(y_true, y_pred_int)
        
        # Datos para curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred_int)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred_int)
        
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

    def visualize_clusters(self, X_test, y_pred, significance):
        """
        Visualiza los clusters y las predicciones del detector de anomalías
        @params
            X_test: np.ndarray
                Conjunto de datos de prueba
            y_pred: np.ndarray
                Predicciones del modelo (1 para anomalía, 0 para normal)
            significance: float
                Nivel de significancia utilizado
        """
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'Clusters y Anomalías (Nivel de Significancia = {significance})')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.savefig(f'results/cp_clusters_sig_{significance}.png')
        plt.close()

def plot_roc_curve(fpr, tpr, auc_roc, significance):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC (Nivel de Significancia = {significance})')
    plt.legend(loc="lower right")
    plt.savefig(f'results/cp_roc_curve_sig_{significance}.png')
    plt.close()

def plot_confusion_matrix(cm, significance):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión (Nivel de Significancia = {significance})')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(f'results/cp_confusion_matrix_sig_{significance}.png')
    plt.close()

def plot_metrics_summary(metrics_list, significances):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    data = {metric: [m[metric] for m in metrics_list] for metric in metrics}
    
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
    plt.savefig('results/cp_metrics_summary.png')
    plt.close()

def main():
    # Crear la carpeta 'results' si no existe
    if not os.path.exists('results'):
        os.makedirs('results')

    np.random.seed(123432) # set seed for reproducibility
    data_generator = DataGenerator(num_samples_per_class=25) # create 10 classes each with 25 samples
    k_nearest_neighbor = KNearestNeighbors(k=10) # Initialize the ICM that uses k-nearest neighbors(k=10)
    conformal_predictor = ConformalAnomalyDetector(ICM=k_nearest_neighbor,z=(data_generator.x,data_generator.y)) # initialize CAD
    
    # Generar datos normales y anómalos
    normal_data = data_generator.x
    anomalies = data_generator.create_anomaly(200) # Generate 200 anomalies
    
    # Crear etiquetas (0 para normal, 1 para anomalía)
    y_normal = np.zeros(normal_data.shape[0])
    y_anomaly = np.ones(anomalies.shape[0])
    
    # Combinar datos y etiquetas
    X = np.vstack((normal_data, anomalies))
    y = np.hstack((y_normal, y_anomaly))
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    significances = [0.025, 0.05, 0.25, 0.5] # see how different significance levels affect results
    all_metrics = []
    for significance in significances:
        conformal_predictor.set_significance(significance) # change significance
        
        # Evaluar el modelo
        metrics = conformal_predictor.evaluate(X_test, y_test)
        all_metrics.append(metrics)
        
        print(f"\nMétricas para nivel de significancia {significance}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Generar y guardar gráficos
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc_roc'], significance)
        plot_confusion_matrix(metrics['confusion_matrix'], significance)

        # Visualizar clusters y anomalías
        y_pred = conformal_predictor(X_test)
        conformal_predictor.visualize_clusters(X_test, y_pred, significance)

    # Generar gráfico de resumen de métricas
    plot_metrics_summary(all_metrics, significances)

if __name__ == '__main__':
    main()