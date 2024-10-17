import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats

# 1. Cargar los datos desde el archivo y tratar valores vacíos o no definidos como NaN
df = pd.read_excel('Dataset_Ventas_Taller.xlsx', na_values=["", " ", "NULL"])

# 2. Eliminar columnas sin nombre (Unnamed)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 3. Mostrar los datos faltantes antes de la limpieza
print("Datos faltantes antes de la limpieza:\n", df.isnull().sum())

# 4. Corregir valores de 'Ventas' que sean inválidos (###)
df['Ventas'] = pd.to_numeric(df['Ventas'], errors='coerce')
df = df[df['Ventas'] >  0]

# 5. Reemplazar precios -999 con NaN
df['Precio'] = df['Precio'].replace(-999, pd.NA)

df = df[df['EdadCliente'] != -1]

# 6. Eliminar filas con cualquier valor faltante en cualquier columna
df = df.dropna()

# 7. Mostrar las primeras filas del DataFrame actualizado
print("\nPrimeras filas después de la limpieza:\n", df.head())

# 8. Mostrar los datos faltantes después de la limpieza
print("\nDatos faltantes después de la limpieza:\n", df.isnull().sum())

# 9. Identificar columnas categóricas y numéricas nuevamente después de la limpieza
categorical_cols = df.select_dtypes(include='object').columns
numerical_cols = df.select_dtypes(exclude='object').columns

print("\nColumnas categóricas:", categorical_cols)
print("Columnas numéricas:", numerical_cols)

# 10. Descripción estadística para columnas numéricas
print("\nDescripción estadística de columnas numéricas:\n", df[numerical_cols].describe())

# 11. Asegurarse de que numerical_cols sea una lista correcta de columnas numéricas válidas
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 12. Boxplot para detectar valores atípicos en las variables numéricas (Matplotlib)
plt.figure(figsize=(10, 6))
df.boxplot(column=numerical_cols)
plt.title('Boxplots de Variables Numéricas')
plt.xticks(rotation=45)
plt.show()

# 13. Histograma de las ventas para ver la distribución (Seaborn)
plt.figure(figsize=(8, 6))
sns.histplot(df['Ventas'], kde=True)
plt.title('Distribución de las Ventas')
plt.xlabel('Ventas')
plt.ylabel('Frecuencia')
plt.show()

# 14. Gráfico de barras para ventas por región (Seaborn)
plt.figure(figsize=(8, 6))
sns.barplot(x='Region', y='Ventas', data=df)
plt.title('Cantidad de Ventas por Región')
plt.xticks(rotation=45)
plt.show()

# 15. Mapa de calor de correlaciones entre las variables numéricas (Seaborn)
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# 16. Gráfico interactivo de ventas por región (Plotly)
fig = px.bar(df, x='Region', y='Ventas', title='Ventas por Región (Interactivo)')
fig.show()

# 17. Gráfico de dispersión interactivo de Precio vs Ventas (Plotly)
fig2 = px.scatter(df, x='Precio', y='Ventas', color='Categoria', 
                  title='Relación Precio vs Ventas por Categoría', 
                  labels={'Precio': 'Precio del Producto', 'Ventas': 'Ventas'})
fig2.show()

# --- Parte de ajuste del modelo ---

# 18. Definir las variables independientes (Precio, Descuento, Edad del Cliente)
X = df[['Precio', 'Descuento', 'EdadCliente']]  # Variables independientes
y = df['Ventas']  # Variable dependiente (Ventas)

# 19. Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 20. Verificar multicolinealidad con el VIF (excluyendo columnas no numéricas)
X_vif = X.select_dtypes(include=[np.number])  # Solo variables numéricas

# Añadir una constante para el cálculo del VIF
X_vif = sm.add_constant(X_vif)  

# Calcular VIF para cada variable numérica
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif["Variable"] = X_vif.columns
print("\nValores de VIF:\n", vif)

# 21. Aplicar Regresión Lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# 22. Evaluar el modelo
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'\nR-cuadrado del modelo de regresión lineal: {r2}')
print(f'RMSE del modelo de regresión lineal: {rmse}')

# 23. Análisis de residuos
residuos = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuos)
plt.axhline(0, color='red', linestyle='--')
plt.title('Análisis de Residuos')
plt.xlabel('Ventas Reales')
plt.ylabel('Residuos')
plt.show()

# 24. Gráfico Q-Q para los residuos
plt.figure(figsize=(8, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Gráfico Q-Q de los Residuos')
plt.show()

# 25. Probar con un modelo polinomial de grado 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
modelo_poly = LinearRegression()
modelo_poly.fit(X_poly, y_train)

# 26. Evaluar el modelo polinomial
y_pred_poly = modelo_poly.predict(poly.transform(X_test))
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = mean_squared_error(y_test, y_pred_poly, squared=False)
print(f'\nR-cuadrado del modelo polinomial: {r2_poly}')
print(f'RMSE del modelo polinomial: {rmse_poly}')

# 27. Probar con un modelo de Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 28. Evaluar el modelo de Random Forest
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'\nR-cuadrado del modelo Random Forest: {r2_rf}')
print(f'RMSE del modelo Random Forest: {rmse_rf}')