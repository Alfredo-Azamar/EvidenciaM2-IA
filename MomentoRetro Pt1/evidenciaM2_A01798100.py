# Author: Alfredo Azamar López
# Profe: Jorge Adolfo Ramírez Uresti
# Evidencia: Implementación de una técnica de aprendizaje máquina sin el uso de un framework
# Algoritmo: Hebb Learning

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
    
    # Se itera sobre cada conjunto de entradas y su correspondiente salida esperada
    for i in range(len(inputs[0])):
        x = [1] + [inputs[j][i] for j in range(num_inputs)]  # Incluye x0 = 1 para el sesgo
        y = output[i]
        
        # Actualización de pesos utilizando la regla de Hebb
        for j in range(len(weights)):
            weights[j] += x[j] * y # Fórmula de Hebb para actualizar los pesos
    
    # Generar la ecuación final
    equation = f"y = {weights[0]}"
    for i in range(1, len(weights)):
        equation += f" + ({weights[i]} * x{i})"
    
    return weights, equation


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
weights, equation = hebb_learning(num_inputs, inputs, output, init_weights, bias_value)

# Mostrar resultados
print("Pesos finales:", weights)
print("Ecuación final:", equation)
