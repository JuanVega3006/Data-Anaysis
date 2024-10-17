import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing

# ===========================
# 1. Descargar y guardar el Adult Income Dataset
# ===========================
# Cargar el dataset
adult = fetch_openml('adult', version=2, as_frame=True)

# Convertir a DataFrame
X_adult = adult.data
y_adult = adult.target
df_adult = pd.concat([X_adult, y_adult], axis=1)

# Guardar en archivo .txt
df_adult.to_csv('adult_income_dataset.txt', sep='\t', index=False)
print("Adult Income Dataset guardado como 'adult_income_dataset.txt'")

# ===========================
# 2. Descargar y guardar el California Housing Dataset
# ===========================
# Cargar el dataset
housing_data = fetch_california_housing()

# Convertir a DataFrame
df_housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df_housing['MedHouseVal'] = housing_data.target  # Añadir el valor de la casa

# Guardar en archivo .txt
df_housing.to_csv('california_housing_dataset.txt', sep='\t', index=False)
print("California Housing Dataset guardado como 'california_housing_dataset.txt'")
