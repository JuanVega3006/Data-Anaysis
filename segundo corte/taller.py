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
df = pd.read_excel('BaseDe DatosFiltrada3.xlsx', na_values=["", " ", "NULL"])

# Leer el archivo Excel y cargarlo en un DataFrame
print("Datos cargados correctamente.")
print(df.head())  # Mostrar las primeras 5 filas


