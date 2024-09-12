# Author: Alfredo Azamar López
# Profe: Jorge Adolfo Ramírez Uresti
# Evidencia: Análisis y Reporte sobre el desempeño del modelo
# Algoritmo: Random Forest

## Importar librerías
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve

## Dataset
data = pd.read_csv("train.csv")
data_desc = data[["HomePlanet","CryoSleep","Destination","Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Transported"]]
print("\nDataset original:")
print(data_desc.head())

## Técnicas de regularización (Normalización de los datos)
# Si tiene na significa que no tiene
data_desc['RoomService'] = data_desc['RoomService'].fillna(0).astype(int)
data_desc['FoodCourt'] = data_desc['FoodCourt'].fillna(0).astype(int)
data_desc['ShoppingMall'] = data_desc['ShoppingMall'].fillna(0).astype(int)
data_desc['Spa'] = data_desc['Spa'].fillna(0).astype(int)
data_desc['VRDeck'] = data_desc['VRDeck'].fillna(0).astype(int)
#se rellena porque si no escribio significa que no tiene VIP
data_desc['VIP'] = data_desc['VIP'].fillna("False").astype(str)
#se rellena con la moda
data_desc['Destination'] = data_desc['Destination'].fillna(data_desc['Destination'].mode()[0]).astype(str)
# Rellenar los valores faltantes con 'Earth'
data_desc['HomePlanet'].fillna('Earth', inplace=True)
# Reemplazar los valores 'Europa' con 'Earth'
data_desc['HomePlanet'] = data_desc['HomePlanet'].replace('Europa', 'Earth')
data_desc['Age'] = data_desc['Age'].fillna(0).astype(int)

## Agrupación de las columnas e implementación de dummy variables
data_desc["Total"] = data_desc[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
data_desc.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)

X_data_desc = data_desc.drop('Transported', axis=1)
y_data_desc = data_desc['Transported']
print("\nDataset preprocesado:")
print(X_data_desc.head())

X_data_desc_encoded = pd.get_dummies(X_data_desc, columns=['VIP'],dtype=int,drop_first=True)
X_data_desc_encoded = pd.get_dummies(X_data_desc_encoded, columns=['HomePlanet'],dtype=int,drop_first=True)
X_data_desc_encoded = pd.get_dummies(X_data_desc_encoded, columns=['Destination'],dtype=int,drop_first=True)
print("\nDataset preprocesado y codificado:")
print(X_data_desc_encoded.head())


## Técnicas de regularización (Escalamiento de los datos)
# scaler = MinMaxScaler()
# X_data_desc_encoded = pd.DataFrame(scaler.fit_transform(X_data_desc_encoded),columns=X_data_desc_encoded.columns)

## Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_data_desc_encoded, y_data_desc, test_size=0.2, random_state=42)
print("\nDatos de entrenamiento:")
print(X_train.head())

print("\nDatos de prueba:")
print(X_test.head())

## Técnicas de regularización (Búsqueda de hiperparámetros, utilizando GridSearchCV)
# Definición de los hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200], # Número de árboles en el bosque
    'max_depth':  [2, 10, 20, 30], # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10], # Número mínimo de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4] # Número mínimo de muestras requeridas en cada hoja
}

## Crear el modelo base
rfBase = RandomForestClassifier(random_state=42)
gridSearch = GridSearchCV(rfBase, param_grid, cv=5) # 5-fold cross-validation

## Entrenamiento del modelo
gridSearch.fit(X_train, y_train)
# rfBase.fit(X_train, y_train)

# Se obtienen los mejores parámetros y el mejor estimador
bestParams = gridSearch.best_params_
print(f"\nMejores parámetros: {bestParams}")

bestEstimator = gridSearch.best_estimator_
print(f"\nMejor estimador: {bestEstimator}")

## Predicción
y_pred = bestEstimator.predict(X_test)
# y_pred = rfBase.predict(X_test)

## Métricas
# -- Exactitud --
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud: {accuracy}")

# -- Reporte de clasificación --
class_report = classification_report(y_test, y_pred)
print("\nReporte de clasificación:")
print(class_report)


## Gráficas

# ++ Graficar exactitud en entrenamiento vs prueba ++
# Exactitud en entrenamiento y prueba
train_accuracy = rfBase.score(X_train, y_train)
test_accuracy = rfBase.score(X_test, y_test)

labels = ['Entrenamiento', 'Prueba']
accuracy_scores = [train_accuracy, test_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(labels, accuracy_scores, color=['blue', 'orange'])
plt.ylabel('Exactitud')
plt.title('Exactitud en Entrenamiento vs Prueba')
plt.ylim(0.1, 1)
plt.show(block=False)
plt.pause(1)

# Curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(rfBase, X_train, y_train, cv=6, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='green', label='Entrenamiento')
plt.plot(train_sizes, test_mean, 'o-', color='darkred', label='Prueba')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Exactitud')
plt.title('Curva de Aprendizaje - Entrenamiento vs Prueba')
plt.legend(loc='best')
plt.show(block=False)
plt.pause(1)



# ++ Análisis del grado de sesgo (bias) ++
# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No transportado', 'Transportado'],
            yticklabels=['No transportado', 'Transportado'])
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de confusión - Conjunto de Prueba')
plt.show(block=False)
plt.pause(1)

# Curva de precisión y recall
# Probabilidades
y_probs = rfBase.predict_proba(X_test)[:, 1]

# Curva de precisión y recall
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Random Forest', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva de Precisión-Recall')
plt.legend()
plt.show(block=False)
plt.pause(1)



# ++ Analizar del grado de varianza ++
# Calcular la predicción en el conjunto de entrenamiento
y_train_pred = rfBase.predict(X_train)

# Métricas para entrenamiento
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.bar(['Entrenamiento', 'Prueba'], [train_accuracy, test_accuracy], color=['olive', 'brown'])
plt.ylabel('Exactitud')
plt.title('Exactitud en Entrenamiento vs Prueba (Varianza)')
plt.ylim(0.1, 1) 
plt.show(block=False)
plt.pause(1)

# Error en entrenamiento y prueba
train_error = 1 - train_accuracy
test_error = 1 - test_accuracy

plt.figure(figsize=(6, 4))
plt.bar(['Entrenamiento', 'Prueba'], [train_error, test_error], color=['olive', 'brown'])
plt.ylabel('Error')
plt.title('Error en Entrenamiento vs Prueba (Varianza)')
plt.ylim(0.0, 0.35)  # Ajustar límites
plt.show(block=False)
plt.pause(1)


# ++ Nivel de ajuste del modelo ++
scores = cross_val_score(rfBase, X_train, y_train, cv=5)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), scores, marker='o', color='magenta')
plt.xlabel('Número de Validaciones (k)')
plt.ylabel('Exactitud')
plt.title('Curva de Validación - Random Forest')
plt.ylim(0.4, 1)  # Ajuste de límites en Y
plt.show(block=False)
plt.pause(1)

# Curva de aprendizaje

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='purple', label='Entrenamiento')
plt.plot(train_sizes, test_mean, 'o-', color='darkcyan', label='Prueba')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Exactitud')
plt.title('Curva de Aprendizaje - Entrenamiento vs Prueba')
plt.legend(loc='best')
plt.show()