import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Cargar datos
@st.cache_data
def load_data():
    xls = pd.ExcelFile("Monitoreo.xlsx")
    df = xls.parse("Sheet1")
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Convertir variables numéricas que puedan haberse leído como texto
    variables_numericas = [
        'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
        'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)',
        'Ruido (dB)', 'UV', 'Humedad (%)', 'Presion (Pa)', 'Temperatura (C)'
    ]
    for var in variables_numericas:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    return df

df = load_data()

st.title("✨ Dashboard de Monitoreo Ambiental")

# Filtros
with st.sidebar:
    st.header("🔍 Filtros")
    variable = st.selectbox("Variable a visualizar", options=[
        'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
        'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)',
        'Ruido (dB)', 'UV', 'Humedad (%)', 'Presion (Pa)', 'Temperatura (C)'
    ])
    fecha = st.date_input("Rango de fechas", [df['Fecha'].min().date(), df['Fecha'].max().date()])

# Validar selección de fechas
if len(fecha) == 2:
    df_filtrado = df[(df['Fecha'].dt.date >= fecha[0]) &
                     (df['Fecha'].dt.date <= fecha[1])]

    # Gráfico de línea
    st.subheader(f"Evolución temporal de {variable}")
    fig_linea = px.line(df_filtrado, x='Fecha', y=variable, markers=True)
    st.plotly_chart(fig_linea, use_container_width=True)

    # Mapa
    st.subheader(f"Ubicación geográfica de registros de {variable}")
    df_mapa = df_filtrado.dropna(subset=[variable])

    # Filtrar solo valores positivos para evitar errores en 'size'
    df_mapa = df_mapa[df_mapa[variable] >= 0]

    fig_mapa = px.scatter_mapbox(
        df_mapa,
        lat="LATITUDE",
        lon="LONGITUDE",
        color=variable,
        size=variable,
        color_continuous_scale="Turbo",
        zoom=10,
        height=500,
        mapbox_style="open-street-map"
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

    # Estadísticas
    st.subheader(f"📊 Estadísticas descriptivas de {variable}")
    st.write(df_filtrado[variable].describe())

    # Descargar
    st.subheader("📁 Descargar datos filtrados")

    @st.cache_data
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Filtrados')
        return output.getvalue()

    datos_excel = to_excel(df_filtrado)
    st.download_button(
        label="🗂️ Descargar Excel",
        data=datos_excel,
        file_name="datos_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("Por favor, selecciona un rango de fechas válido.")
