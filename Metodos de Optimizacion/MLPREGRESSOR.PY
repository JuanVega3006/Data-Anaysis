from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el dataset
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo MLPRegressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_reg.fit(X_train, y_train)

# Realizar predicciones
y_pred = mlp_reg.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R2 Score: {r2:.4f}')

