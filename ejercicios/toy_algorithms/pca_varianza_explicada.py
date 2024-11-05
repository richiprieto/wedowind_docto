#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2024 Eón Corp

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use('Qt5Agg')

# Cargar un conjunto de datos (en este caso, el conjunto de datos Iris)
data = load_iris()
X = data.data

# Estandarizar los datos
X_scaled = StandardScaler().fit_transform(X)

# Aplicar PCA
pca = PCA()
pca.fit(X_scaled)

# Obtener la varianza explicada por cada componente
varianza = pca.explained_variance_ratio_

# Calcular la varianza acumulada
var_acum = np.cumsum(varianza)

# Visualizar la varianza explicada y la varianza acumulada
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianza) + 1), varianza, alpha=0.6, label='Varianza Explicada')
plt.plot(range(1, len(var_acum) + 1), var_acum, marker='o', color='r', label='Varianza Acumulada')
plt.title('Varianza Explicada y Acumulada por Componentes Principales')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Proporción de Varianza')
plt.xticks(range(1, len(varianza) + 1))
plt.axhline(y=0.90, color='g', linestyle='--', label='90% Varianza Acumulada')
plt.legend()
plt.grid()
plt.show()
