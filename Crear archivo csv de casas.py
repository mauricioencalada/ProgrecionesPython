# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:57:29 2023

@author: Diego Encalada
"""
import pandas as pd
import numpy as np

# Generar datos sintéticos para características de casas
num_samples = 25

# Generar valores no negativos para el número de habitaciones y tamaño del lote
habitaciones = np.random.exponential(2, num_samples)+3
tamaño_lote = np.random.exponential(100, num_samples)
ubicacion = np.random.choice(['centro', 'suburbio', 'rural'], size=num_samples)
#otra_caracteristica = np.random.normal(0, 1, num_samples)  # Ejemplo de otra característica

# Crear un DataFrame de pandas con las características generadas
data = pd.DataFrame({
    'habitaciones': habitaciones,
    'tamaño_lote': tamaño_lote,
    'ubicacion': ubicacion,
    #'otra_caracteristica': otra_caracteristica
})

# Asegurar que el número de habitaciones y tamaño del lote sean no negativos
data['habitaciones'] = data['habitaciones'].apply(lambda x: max(int(x), 0))
data['tamaño_lote'] = data['tamaño_lote'].apply(lambda x: max(int(x), 0))

# Generar precios de venta (variable objetivo)
data['precio_venta'] = np.abs(np.round(np.random.uniform(100000, 1000000, num_samples), 2))

# Guardar el DataFrame en un archivo CSV
data.to_csv('house_data.csv', index=False)
