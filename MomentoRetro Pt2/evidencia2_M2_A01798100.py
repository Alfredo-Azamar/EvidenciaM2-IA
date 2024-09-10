# Author: Alfredo Azamar López
# Profe: Jorge Adolfo Ramírez Uresti
# Evidencia: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución
# Algoritmo: Random Forest

# Importar librerías
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definición de los hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200], # Número de árboles en el bosque
    'max_depth':  [2, 10, 20, 30], # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10], # Número mínimo de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4] # Número mínimo de muestras requeridas en cada hoja
}

# Crear el modelo base
rfBase = RandomForestClassifier(random_state=42)
gridSearch = GridSearchCV(rfBase, param_grid, cv=5) # 5-fold cross-validation

# Entrenamiento del modelo
gridSearch.fit(X_train, y_train)

# Se obtienen los mejores parámetros y el mejor estimador
bestParams = gridSearch.best_params_
print(f"\nMejores parámetros: {bestParams}")

bestEstimator = gridSearch.best_estimator_
print(f"\nMejor estimador: {bestEstimator}")

# Predicción
y_pred = bestEstimator.predict(X_test)

# Métricas
# -- Exactitud --
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud: {accuracy}")

# -- Reporte de clasificación --
class_report = classification_report(y_test, y_pred)
print("\nReporte de clasificación:")
print(class_report)

# -- Matriz de confusión --
confusion_matrix = confusion_matrix(y_test, y_pred)

# Graficarlo
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de confusión - Random Forest')
plt.show()