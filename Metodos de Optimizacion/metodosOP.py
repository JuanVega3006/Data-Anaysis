import numpy as np

# Definimos la función objetivo y su gradiente (derivada)
def function(x):
    return x**2

def gradient(x):
    return 2*x

# Parámetros generales
x_init = 10  # Valor inicial
alpha = 0.1  # Tasa de aprendizaje
epochs = 100  # Número de iteraciones
epsilon = 1e-8  # Pequeña constante para evitar divisiones por cero

# 1. Gradiente Descendente
def gradient_descent(x):
    print("\n--- Gradiente Descendente ---")
    for i in range(epochs):
        grad = gradient(x)
        x = x - alpha * grad
        print(f"Epoch {i+1}: x = {x}, f(x) = {function(x)}")
    return x

# 2. Gradiente Descendente con Momentum
def gradient_descent_momentum(x):
    print("\n--- Gradiente Descendente con Momentum ---")
    v = 0  # Inicialización de la velocidad
    beta = 0.9  # Parámetro de momentum
    for i in range(epochs):
        grad = gradient(x)
        v = beta * v + (1 - beta) * grad  # Acumulamos el gradiente con momentum
        x = x - alpha * v
        print(f"Epoch {i+1}: x = {x}, f(x) = {function(x)}")
    return x

# 3. Adagrad
def adagrad(x):
    print("\n--- Adagrad ---")
    grad_sum = 0  # Inicialización del acumulador de gradientes
    for i in range(epochs):
        grad = gradient(x)
        grad_sum += grad**2
        x = x - (alpha / (np.sqrt(grad_sum) + epsilon)) * grad
        print(f"Epoch {i+1}: x = {x}, f(x) = {function(x)}")
    return x

# 4. RMSProp
def rmsprop(x):
    print("\n--- RMSProp ---")
    grad_squared = 0  # Inicialización del acumulador de gradientes al cuadrado
    beta = 0.9  # Parámetro decay rate
    for i in range(epochs):
        grad = gradient(x)
        grad_squared = beta * grad_squared + (1 - beta) * grad**2
        x = x - (alpha / (np.sqrt(grad_squared) + epsilon)) * grad
        print(f"Epoch {i+1}: x = {x}, f(x) = {function(x)}")
    return x

# 5. Adam
def adam(x):
    print("\n--- Adam ---")
    m = 0  # Inicialización del momento
    v = 0  # Inicialización de la varianza
    beta1 = 0.9  # Decay rate para el momento
    beta2 = 0.999  # Decay rate para la varianza
    for i in range(1, epochs+1):
        grad = gradient(x)
        m = beta1 * m + (1 - beta1) * grad  # Calcular el momento
        v = beta2 * v + (1 - beta2) * grad**2  # Calcular la segunda estimación
        m_hat = m / (1 - beta1**i)  # Corregir sesgo para el momento
        v_hat = v / (1 - beta2**i)  # Corregir sesgo para la varianza
        x = x - (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
        print(f"Epoch {i}: x = {x}, f(x) = {function(x)}")
    return x

# Ejecutamos cada método de optimización
x = x_init
x = gradient_descent(x)

x = x_init
x = gradient_descent_momentum(x)

x = x_init
x = adagrad(x)

x = x_init
x = rmsprop(x)

x = x_init
x = adam(x)
