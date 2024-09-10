# Author: Alfredo Azamar López
# Profe: Jorge Adolfo Ramírez Uresti
# Evidencia: Implementación de una técnica de aprendizaje máquina sin el uso de un framework
# Algoritmo: Hebb Learning

# Importar librerías
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Función para el aprendizaje de Hebb
def hebb_learning(num_inputs, inputs, output, init_weights, bias_value):
    """
    Parámetros:
    - num_inputs: Número de entradas o características (x1, x2, ..., xn).
    - inputs: Lista de listas con los valores de las entradas (x1, x2, ..., xn).
    - output: Lista con los valores de salida esperados.
    - init_weights: Valor inicial de los pesos.
    - bias_value: Valor inicial del sesgo.

    Regresa:
    - weights: Lista con los pesos finales.
    - equation: Ecuación final que representa la función aprendida.
    """

    # Inicializar pesos, incluyendo el peso del sesgo (w0)
    weights = [init_weights] * (num_inputs + bias_value)  # Incluye el peso w0 para el sesgo
    
    # Lista para almacenar las predicciones
    predictions = []

    # Se itera sobre cada conjunto de entradas y su correspondiente salida esperada
    for i in range(len(inputs[0])):
        x = [1] + [inputs[j][i] for j in range(num_inputs)]  # Incluye x0 = 1 para el sesgo
        y = output[i]

        # Predicción
        prediction = 1 if sum([x[j] * weights[j] for j in range(len(weights))]) >= 0 else -1
        predictions.append(prediction)
        
        # Actualización de pesos utilizando la regla de Hebb
        for j in range(len(weights)):
            weights[j] += x[j] * y # Fórmula de Hebb para actualizar los pesos
    
    # Generar la ecuación final
    equation = f"y = {weights[0]}"
    for i in range(1, len(weights)):
        equation += f" + ({weights[i]} * x{i})"

    # Calcular métricas
    accuracy = accuracy_score(output, predictions)
    precision = precision_score(output, predictions, average='binary', pos_label=1)
    recall = recall_score(output, predictions, average='binary', pos_label=1)
    f1 = f1_score(output, predictions, average='binary', pos_label=1)
    conf_matrix = confusion_matrix(output, predictions)
    
    return weights, equation, accuracy, precision, recall, f1, conf_matrix


# Ejemplo de uso
num_inputs = 2

# Ejemplo 1
inputs = [
    [-1, -1, 1, 1],  # x1
    [-1, 1, -1, 1]   # x2
]
output = [-1,-1,-1,1]

# Ejemplo 2
inputs2 = [
    [-1, 1, 1, 1],  # x1
    [1, -1, -1, -1]   # x2
]
output2 = [-1,-1,-1,-1]

# Ejemplo 3
inputs3 = [
    [1, -1, 1, -1],  # x1
    [1, -1, 1, 1]   # x2
]
output3 = [1,-1,1,-1]

# Valores iniciales
init_weights = 0
bias_value = 1

# Ejecutar el algoritmo
weights, equation, accuracy, precision, recall, f1, conf_matrix = hebb_learning(num_inputs, inputs, output, init_weights, bias_value)

# Mostrar resultados
print("\nPesos finales:", weights)
print("\nEcuación final:", equation)

print("\n~Métricas~")
print(f"Exactitud: {accuracy}")
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("\nMatriz de confusión:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de confusión - Algoritmo de Hebb')
plt.show()