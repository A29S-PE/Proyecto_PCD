import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard", layout="wide")

@st.cache_resource
def load_dataset():
    df = pd.read_excel("Monitoreo.xlsx", na_values=['-','',0])
    df['Date'] = df.Fecha.dt.date
    df['Hour'] = df.Fecha.dt.hour
    condiciones = [
        ((df.LATITUDE == -12.119134) & (df.LONGITUDE == -77.028848)),
        ((df.LATITUDE == -12.109723) & (df.LONGITUDE == -77.051940)),
        ((df.LATITUDE == -12.072736) & (df.LONGITUDE == -77.082687)),
        ((df.LATITUDE == -12.040278) & (df.LONGITUDE == -77.043609))
    ]
    valores = ["Óvalo de Miraflores", "Complejo Deportivo Municipal Niño Manuel Bonilla", "PUCP", "Otro"]
    df["Zona"] = np.select(condiciones, valores, default="Desconocido")
    return df

df = load_dataset()

st.title("✨ Dashboard de Monitoreo Ambiental")

# Mostrar dataset
if st.checkbox("Mostrar algunos ejemplos de datos"):
    st.write(df[['Fecha', 'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)',
       'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)',
       'Ruido (dB)', 'UV', 'Humedad (%)', 'LATITUDE', 'LONGITUDE',
       'Presion (Pa)', 'Temperatura (C)']].sample(5))

# Selección de variable para graficar
variables_numericas = ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)',
       'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)',
       'Ruido (dB)', 'UV', 'Humedad (%)', 'Presion (Pa)', 'Temperatura (C)']
columna = st.selectbox("Selecciona una variable para graficar", variables_numericas)

# Selección de zona
variables_zonas = ["Óvalo de Miraflores", "Complejo Deportivo Municipal Niño Manuel Bonilla", "PUCP", "Otro"]
zona = st.selectbox("Selecciona la Zona", variables_zonas)

# Mapa
st.map(df[df.Zona == zona][["LATITUDE", "LONGITUDE"]], height = 200, zoom = 13)

# Línea de tiempo
df_temp = df[df.Zona == zona]
df_temp = df_temp.groupby('Date')[columna].mean().reset_index()
st.subheader(f"📈 Evolución diaria del valor promedio de {columna} en {zona}")
plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_temp["Date"], df_temp[columna], label=columna, color="royalblue", linewidth=2)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel(columna, fontsize=12)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()
# ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
st.pyplot(fig)
