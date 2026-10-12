#
# TAREA 4: SCRIPT DE STREAMLIT
#
#ESCRIBE TU CÓDIGO AQUÍ:
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

#1.Configuración de página
st.set_page_config(page_title="Sistema EcoTránsito CDMX", layout="wide")

#2.Carga y filtrado de datos con barra lateral
df_clean = pd.read_csv("cdmx_air_traffic_clean.csv")

zonas_disponibles = df_clean['Zona_Estacion'].unique()

st.sidebar.header("Filtros")
zonas_seleccionadas = st.sidebar.multiselect(
    "Selecciona la(s) Zona(s):",
    options=zonas_disponibles,
    default=zonas_disponibles
)

df_filtrado = df_clean[df_clean['Zona_Estacion'].isin(zonas_seleccionadas)]

# 3.Métricas principales (KPIs)
col1, col2, col3 = st.columns(3)

total_lecturas = df_filtrado.shape[0]
promedio_pm25 = df_filtrado['Concentracion_PM25'].mean()
max_trafico = df_filtrado['Indice_Trafico_Vehicular'].max()

with col1:
    st.metric(label="Total de lecturas activas", value=total_lecturas)
with col2:
    st.metric(label="Promedio PM2.5 (µg/m³)", value=f"{promedio_pm25:.2f}")
with col3:
    st.metric(label="Máx. Índice de Tráfico", value=f"{max_trafico:.1f}")

# 4.Pestañas interactivas de visualización (Plotly + Seaborn)
tab1, tab2 = st.tabs(["Dispersión PM2.5 vs Tráfico", "Mapa de Calor de Correlación"])

with tab1:
    st.subheader("Relación entre Índice de Tráfico y PM2.5")
    fig_scatter = px.scatter(
        df_filtrado,
        x='Indice_Trafico_Vehicular',
        y='Concentracion_PM25',
        color='Zona_Estacion',
        title="Dispersión por Zona de Estación"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Correlación de Variables Numéricas")
    fig_heatmap, ax = plt.subplots(figsize=(8, 6))

    #Seleccionar solo columnas numéricas para la matriz
    cols_num = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular']
    matriz_corr = df_filtrado[cols_num].corr()

    sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_heatmap)