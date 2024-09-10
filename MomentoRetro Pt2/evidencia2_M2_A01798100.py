# Author: Alfredo Azamar López
# Profe: Jorge Adolfo Ramírez Uresti
# Evidencia: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución
# Algoritmo: Random Forest

# Importar librerías
import numpy as np
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
# model = RandomForestClassifier(n_estimators=100, random_state=42)
rfBase = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':  [2, 10, 20, 30],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]
}

gridSearch = GridSearchCV(rfBase, param_grid, cv=5)

# Entrenamiento
gridSearch.fit(X_train, y_train)

bestParams = gridSearch.best_params_
print(f"Mejores parámetros: {bestParams}")

# Predicción
bestRF = gridSearch.best_estimator_
y_pred = bestRF.predict(X_test)

# Métricas
# -- Accuracy --
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud: {accuracy}")

# -- Matriz de confusión --
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print("Matriz de confusión:")
# print(confusion_matrix)

# -- Reporte de clasificación --
class_report = classification_report(y_test, y_pred)
print("Reporte de clasificación:")
print(class_report)

