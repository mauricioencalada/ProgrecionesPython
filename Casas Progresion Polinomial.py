# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:46:01 2023

@author: Diego 
"""
#progresion polinomial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos desde el archivo CSV
data = pd.read_csv('house_data.csv')

# Separar las características (X) y la variable objetivo (y)
X = data[['habitaciones']]
y = data['precio_venta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión polinómica de grado 2
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predecir en el conjunto de entrenamiento y prueba
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

# Calcular métricas de rendimiento
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'RMSE en entrenamiento: {train_rmse:.2f}')
print(f'RMSE en prueba: {test_rmse:.2f}')
print(f'R² en entrenamiento: {r2_train:.2f}')
print(f'R² en prueba: {r2_test:.2f}')

# Visualizar la regresión polinómica
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='skyblue', label='Datos reales')
plt.scatter(X_train, y_pred_train, color='orange', label='Predicciones en entrenamiento')
plt.scatter(X_test, y_pred_test, color='green', label='Predicciones en prueba')
plt.xlabel('Número de Habitaciones')
plt.ylabel('Precio de Venta')
plt.title('Regresión Polinómica de Grado 2')
plt.legend()
plt.show()