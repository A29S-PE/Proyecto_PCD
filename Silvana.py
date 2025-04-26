import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_excel("Monitoreo.xlsx")
    df.rename(columns={"PM2,5 (ug/m3)": "PM2.5 (ug/m3)"}, inplace=True)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    variables_numericas = [
        'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
        'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)',
        'Ruido (dB)', 'UV', 'Humedad (%)', 'Presion (Pa)', 'Temperatura (C)'
    ]
    for var in variables_numericas:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    df['Hora'] = df['Fecha'].dt.hour
    df['Día'] = df['Fecha'].dt.date
    return df


df = load_data()

st.title("🌍 Análisis de Contaminación por Horas")

# --- Filtros ---
with st.sidebar:
    st.header("🔍 Filtros")
    contaminante = st.selectbox("Selecciona un contaminante", [
        'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)', 'CO (ug/m3)', 'O3 (ug/m3)'
    ])

# --- Tabs de visualización ---
tabs = st.tabs(["Promedio Horario", "Boxplot por Hora", "Mapa Animado", "Heatmap Hora vs Día"])

# --- Promedio horario ---
with tabs[0]:
    st.subheader(f"Promedio horario de {contaminante}")
    promedio_hora = df.groupby('Hora')[contaminante].mean().reset_index()
    fig = px.line(promedio_hora, x='Hora', y=contaminante, markers=True)
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

# --- Boxplot por hora ---
with tabs[1]:
    st.subheader(f"Distribución de {contaminante} por hora")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Hora', y=contaminante, ax=ax)
    ax.set_title(f"Boxplot de {contaminante} por Hora")
    st.pyplot(fig)

# --- Mapa Animado ---
with tabs[2]:
    st.subheader(f"Mapa animado de {contaminante} por hora")
    df_mapa = df.dropna(subset=[contaminante, 'LATITUDE', 'LONGITUDE'])

    # -- Filtrar solo valores positivos
    df_mapa = df_mapa[df_mapa[contaminante] >= 0]

    fig = px.scatter_mapbox(
        df_mapa,
        lat="LATITUDE",
        lon="LONGITUDE",
        color=contaminante,
        size=contaminante,
        animation_frame="Hora",
        color_continuous_scale="inferno",
        size_max=15,
        zoom=11,
        mapbox_style="open-street-map",
        title=f"Concentración de {contaminante} por hora"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# --- Heatmap Hora vs Día ---
with tabs[3]:
    st.subheader(f"Heatmap de {contaminante} por hora y día")
    heatmap_data = df.pivot_table(index='Día', columns='Hora', values=contaminante, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax)
    ax.set_title(f"Matriz de calor de {contaminante}")
    st.pyplot(fig)
