# ==============================================================================
# TAREA 1: INGESTA, DIAGNÓSTICO Y SUSTITUCIÓN DE CENTINELAS[cite: 1]
# ==============================================================================
import numpy as np
import pandas as pd

# 1. Cargar o definir el dataset[cite: 1]
data = {
    'ID_Registro': [
        'REG-001',
        'REG-002',
        'REG-003',
        'REG-004',
        'REG-005',
        'REG-006',
        'REG-007',
        'REG-008',
        'REG-009',
        'REG-010',
        'REG-010',
    ],
    'Timestamp': [
        '2026-10-01 08:00:00',
        '2026-10-01 09:00:00',
        '2026-10-01 10:00:00',
        '2026-10-01 11:00:00',
        '2026-10-01 12:00:00',
        '2026-10-01 13:00:00',
        '2026-10-01 14:00:00',
        '2026-10-01 15:00:00',
        '2026-10-01 16:00:00',
        '2026-10-01 17:00:00',
        '2026-10-01 17:00:00',
    ],
    'Zona_Estacion': [
        'Centro_Historico',
        'centro_historico',
        'CENTRO_HISTORICO',
        'Zona_Oriente',
        'Zona_Oriente',
        'zona_oriente',
        'Zona_Sur',
        'Zona_Sur',
        'Zona_Sur',
        'Zona_Sur',
        'Zona_Sur',
    ],
    'Concentracion_PM25': [
        35.4,
        42.1,
        -999.0,
        120.5,
        np.nan,
        310.0,
        18.2,
        20.1,
        19.5,
        19.5,
        19.5,
    ],
    'Nivel_NO2': [
        42.1,
        48.5,
        51.0,
        89.2,
        82.0,
        195.0,
        22.4,
        -999.0,
        25.1,
        25.1,
        25.1,
    ],
    'Indice_Trafico_Vehicular': [
        85.0,
        92.5,
        88.0,
        98.0,
        91.2,
        15.0,
        45.0,
        42.1,
        np.nan,
        41.0,
        41.0,
    ],
    'Estado_Sensor': [
        'Normal',
        'Normal',
        'Alerta',
        'Alerta',
        'Normal',
        'Falla',
        'Normal',
        'Normal',
        'Normal',
        'Normal',
        'Normal',
    ],
}

df_raw = pd.DataFrame(data)

#ESCRIBE TU CÓDIGO AQUÍ:
# 
#a) Diagnóstico inicial de duplicados y centinelas

#Seleccionar columnas numéricas
columnas_numericas = df_raw.select_dtypes(include=[np.number]).columns

#Contar valores NaN explícitos
conteo_nan = df_raw[columnas_numericas].isna().sum()

#Contar valores centinela -999.0
conteo_centinelas = (df_raw[columnas_numericas] == -999.0).sum()

#Contar filas totalmente duplicadas
filas_duplicadas = df_raw.duplicated().sum()

print('--- DIAGNÓSTICO INICIAL ---')
print('Faltantes explícitos (NaN) por columna num:\n', conteo_nan)
print('\nValores centinela (-999.0) por columna num:\n', conteo_centinelas)
print('\nFilas totalmente duplicadas:', filas_duplicadas)


#b) Reemplazar centinela -999.0 por np.nan y eliminar duplicados

#Reemplazar -999.0 por np.nan
df_raw.replace(-999.0, np.nan, inplace=True)

#Eliminar filas totalmente duplicadas
df_raw.drop_duplicates(inplace=True)

#Visualización del resultado final
print('\n--- DATAFRAME PROCESADO ---')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_raw)
#
# TAREA 2: PIPELINE DE LIMPIEZA, FEATURE ENGINEERING E IMPUTACIÓN
#
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

df_clean = df_raw.copy()

# # ESCRIBE TU CÓDIGO AQUÍ:
#
#1. Homogeneizar 'Zona_Estacion'                                 
# Se estandariza a formato Title Case para corregir inconsistencias de caja
df_clean['Zona_Estacion'] = df_clean['Zona_Estacion'].str.title()

#2. Convertir Timestamp a datetime y extraer Hora_Dia
df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
df_clean['Hora_Dia'] = df_clean['Timestamp'].dt.hour

#3. Detección de outliers por IQR en 'Concentracion_PM25' y sustitución por np.nan
Q1 = df_clean['Concentracion_PM25'].quantile(0.25)
Q3 = df_clean['Concentracion_PM25'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

#Remplazar el outlier de la fila REG-006 (310.0) por np.nan
df_clean.loc[
    df_clean['Concentracion_PM25'] > limite_superior, 'Concentracion_PM25'
] = np.nan

#4. Imputación avanzada k-NN sobre variables numéricas
cols_numericas = [
    'Concentracion_PM25',
    'Nivel_NO2',
    'Indice_Trafico_Vehicular',
]
imputer = KNNImputer(n_neighbors=2)

df_clean[cols_numericas] = imputer.fit_transform(df_clean[cols_numericas])

#Visualización del resultado final
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df_clean)
#
# TAREA 3: EDA Y ANÁLISIS DE CORRELACIÓN
#
# ESCRIBE TU CÓDIGO AQUÍ:
#
#1. Estadísticas de forma para Concentracion_PM25
media_pm25 = df_clean['Concentracion_PM25'].mean()
mediana_pm25 = df_clean['Concentracion_PM25'].median()
std_pm25 = df_clean['Concentracion_PM25'].std()
skew_pm25 = df_clean['Concentracion_PM25'].skew()
kurt_pm25 = df_clean['Concentracion_PM25'].kurt()

print('--- Estadísticas de Concentracion_PM25 ---')
print('Media:', media_pm25)
print('Mediana:', mediana_pm25)
print('Desviación estándar:', std_pm25)
print('Asimetría (skew):', skew_pm25)
print('Curtosis (kurt):', kurt_pm25)


#2. Matrices de correlación Pearson y Spearman
cols_num = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular']

corr_pearson = df_clean[cols_num].corr(method='pearson')
corr_spearman = df_clean[cols_num].corr(method='spearman')

print('\n--- Matriz de Correlación (Pearson) ---')
print(corr_pearson)

print('\n--- Matriz de Correlación (Spearman) ---')
print(corr_spearman)

print('\n--- Pregunta diagnóstica ---')
print("El par de variables con |r| >= 0.85 es 'Concentracion_PM25' y 'Nivel_NO2' (Spearman = 0.9416).")
print("Esto representa el problema de MULTICOLINEALIDAD ALTA.")
print("En un modelo de regresión, esto genera inestabilidad en la estimación de los coeficientes,")
print("infla los errores estándar y dificulta determinar la importancia individual de cada variable.")
