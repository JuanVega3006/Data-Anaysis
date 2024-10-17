import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Función y gradiente
def function(x):
    return x**2

def gradient(x):
    return 2*x

# Parámetros generales
alpha = 0.1  # Tasa de aprendizaje
epochs = 100  # Número de iteraciones
epsilon = 1e-8  # Pequeña constante para evitar divisiones por cero

# Métodos de optimización
def gradient_descent(x):
    xs, fs = [], []
    for i in range(epochs):
        grad = gradient(x)
        x = x - alpha * grad
        xs.append(x)
        fs.append(function(x))
    return xs, fs

def gradient_descent_momentum(x):
    xs, fs = [], []
    v = 0
    beta = 0.9
    for i in range(epochs):
        grad = gradient(x)
        v = beta * v + (1 - beta) * grad
        x = x - alpha * v
        xs.append(x)
        fs.append(function(x))
    return xs, fs

def adagrad(x):
    xs, fs = [], []
    grad_sum = 0
    for i in range(epochs):
        grad = gradient(x)
        grad_sum += grad**2
        x = x - (alpha / (np.sqrt(grad_sum) + epsilon)) * grad
        xs.append(x)
        fs.append(function(x))
    return xs, fs

def rmsprop(x):
    xs, fs = [], []
    grad_squared = 0
    beta = 0.9
    for i in range(epochs):
        grad = gradient(x)
        grad_squared = beta * grad_squared + (1 - beta) * grad**2
        x = x - (alpha / (np.sqrt(grad_squared) + epsilon)) * grad
        xs.append(x)
        fs.append(function(x))
    return xs, fs

def adam(x):
    xs, fs = [], []
    m, v = 0, 0
    beta1, beta2 = 0.9, 0.999
    for i in range(1, epochs+1):
        grad = gradient(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        x = x - (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
        xs.append(x)
        fs.append(function(x))
    return xs, fs

# Función para graficar el método seleccionado
def plot_method(method_name):
    x_init = 10
    if method_name == "Gradiente Descendente":
        xs, fs = gradient_descent(x_init)
    elif method_name == "Gradiente Descendente con Momentum":
        xs, fs = gradient_descent_momentum(x_init)
    elif method_name == "Adagrad":
        xs, fs = adagrad(x_init)
    elif method_name == "RMSProp":
        xs, fs = rmsprop(x_init)
    elif method_name == "Adam":
        xs, fs = adam(x_init)

    # Limpiar la figura actual
    ax.clear()

    # Graficar x vs f(x)
    ax.plot(xs, fs, label=method_name)
    
    # Establecer límites fijos para los ejes
    ax.set_xlim(-10, 10)  # Limites para el eje X
    ax.set_ylim(0, 120)   # Limites para el eje Y

    ax.set_title(f"Optimización usando {method_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()

    # Redibujar en la interfaz
    canvas.draw()

# Crear la ventana principal
root = tk.Tk()
root.title("Visualización de Métodos de Optimización")

# Crear la figura para la gráfica
fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Crear el menú desplegable para seleccionar el método
method_label = tk.Label(root, text="Selecciona un método de optimización:")
method_label.pack(pady=10)

method_options = ["Gradiente Descendente", "Gradiente Descendente con Momentum", "Adagrad", "RMSProp", "Adam"]
method_var = tk.StringVar(value=method_options[0])
method_menu = ttk.Combobox(root, textvariable=method_var, values=method_options, state="readonly")
method_menu.pack()

# Botón para actualizar la gráfica
plot_button = tk.Button(root, text="Graficar", command=lambda: plot_method(method_var.get()))
plot_button.pack(pady=10)

# Iniciar la interfaz
root.mainloop()
