import plotly.express as px
import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
import base64
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Calidad del Aire - Lima, Perú",
    page_icon="🌬️",
    layout="wide"
)

# --- Constantes ---
NOMBRE_ARCHIVO = 'Monitoreo.xlsx'
IMAGEN_ARCHIVO = 'pucp.png'
# Columnas a limpiar (convertir a numérico)
COLUMNAS_A_LIMPIAR = ['H2S (ug/m3)', 'SO2 (ug/m3)']
# Columnas de coordenadas
COL_LATITUD = 'LATITUDE'
COL_LONGITUD = 'LONGITUDE'
COL_ZONA = 'Zona'
COL_FECHA_HORA = 'Fecha'
COL_FECHA = 'Date'
COL_HORA = 'Hour'
COL_ESTACION = 'NOMBRE DE LA UO'
# Definir listas de columnas para facilitar selección
COLUMNAS_CONTAMINANTES = ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2,5 (ug/m3)', 'SO2 (ug/m3)']
COLUMNAS_METEOROLOGICAS = ['Humedad (%)', 'Presion (Pa)', 'Temperatura (C)', 'UV']
COLUMNAS_OTRAS = ['Ruido (dB)']
# Columnas clave para KPIs
COLUMNAS_KPI = COLUMNAS_CONTAMINANTES + COLUMNAS_OTRAS

# --- Funciones ---
@st.cache_resource
def load_dataset(file_path) -> pd.DataFrame:
    """Carga los datos desde el archivo Excel y realiza limpieza básica."""
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la misma carpeta que app.py.")
        st.stop()

    try:
        df = pd.read_excel(file_path, na_values=['-','',0])
        df.columns = df.columns.str.strip()
        
        df[COL_FECHA_HORA] = pd.to_datetime(df[COL_FECHA_HORA], errors='coerce')
        df = df.sort_values(by=COL_FECHA_HORA)

        df['Date'] = df[COL_FECHA_HORA].dt.date
        df['Hour'] = df[COL_FECHA_HORA].dt.hour

        condiciones = [
            ((df[COL_LATITUD] == -12.119134) & (df[COL_LONGITUD] == -77.028848)),
            ((df[COL_LATITUD] == -12.109723) & (df[COL_LONGITUD] == -77.051940)),
            ((df[COL_LATITUD] == -12.072736) & (df[COL_LONGITUD] == -77.082687)),
            ((df[COL_LATITUD] == -12.040278) & (df[COL_LONGITUD] == -77.043609))
        ]
        valores = ["Óvalo de Miraflores", "Complejo Deportivo Municipal Niño Manuel Bonilla", "Pontificia Universidad Católica del Perú", "Enrique Meiggs con Alfonso Ugarte"]
        df["Zona"] = np.select(condiciones, valores, default="Desconocido")

        for col in COLUMNAS_A_LIMPIAR:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    st.warning(f"Advertencia: No se encontró la columna '{col}' para limpieza.")

        if COL_LATITUD not in df.columns or COL_LONGITUD not in df.columns:
                st.warning(f"Advertencia: No se encontraron las columnas '{COL_LATITUD}' y/o '{COL_LONGITUD}'. El mapa no funcionará.")

        columnas_numericas_potenciales = COLUMNAS_CONTAMINANTES + COLUMNAS_METEOROLOGICAS + COLUMNAS_OTRAS
        for col in columnas_numericas_potenciales:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    
    except Exception as e:
        st.error(f"Ocurrió un error al cargar o procesar el archivo Excel: {e}")
        st.info("Asegúrate de que el archivo no esté corrupto y sea un .xlsx válido.")
        st.info("Puede que necesites instalar 'openpyxl': pip install openpyxl")
        st.stop()

@st.cache_resource
def load_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_data
def to_excel(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtrados')
    return output.getvalue()


# --- Carga de Datos ---
df_original = load_dataset(NOMBRE_ARCHIVO)

# Estas son las columnas que se podrán seleccionar en los gráficos
columnas_numericas_disponibles = [
    col for col in (COLUMNAS_CONTAMINANTES + COLUMNAS_METEOROLOGICAS + COLUMNAS_OTRAS)
    if col in df_original.columns and pd.api.types.is_numeric_dtype(df_original[col])
]

df = df_original.copy()

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Navegación")


fecha_inicio = None
fecha_fin = None
df_filtrado = df.copy()

df_valid_dates = df.dropna(subset=[COL_FECHA]).copy()

fecha_min    = df_filtrado[COL_FECHA].min()
fecha_max    = df_filtrado[COL_FECHA].max()
date_range   = st.sidebar.date_input("Seleccione el rango de fechas:", [fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max)
if len(date_range) == 2:
    fecha_inicio = pd.to_datetime(date_range[0]).date()
    fecha_fin    = pd.to_datetime(date_range[1]).date()
    df_filtrado = df_valid_dates[(df_valid_dates[COL_FECHA] >= fecha_inicio) & (df_valid_dates[COL_FECHA] <= fecha_fin)].copy()
elif len(date_range) == 1:
    fecha_inicio = pd.to_datetime(date_range[0]).date()
    fecha_fin    = pd.to_datetime(date_range[0]).date()
    df_filtrado = df_valid_dates[(df_valid_dates[COL_FECHA] >= fecha_inicio) & (df_valid_dates[COL_FECHA] <= fecha_fin)].copy()


opciones_zonas = ["Todas",
                "Óvalo de Miraflores", 
                "Complejo Deportivo Municipal Niño Manuel Bonilla", 
                "Pontificia Universidad Católica del Perú", 
                "Otro"]
zona_seleccionada = st.sidebar.selectbox("Selecciona la estación:", opciones_zonas)
if zona_seleccionada != "Todas":
    df_filtrado = df_filtrado[(df_filtrado[COL_ZONA] == zona_seleccionada)].copy()


# --- Menú Principal ---
opciones_menu = [
    "📍 Resumen General",
    "🧪📈 Tendencia: Contaminantes",
    "🧪📊 Comparativa: Contaminantes",
    "🌦️📈 Tendencia: D. Meteorológicos",
    "🔗 Correlaciones",
    "ℹ️ Información Técnica del Dataset"
]
# Asegurarnos de que el índice 0 siempre sea Resumen General
seleccion = st.sidebar.radio("Selecciona una sección:", opciones_menu, index=0)

base64_img = load_image(IMAGEN_ARCHIVO)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{base64_img}' width='100'/>
        <p style='font-size: 12px; margin-top: 5px;'>Pontificia Universidad Católica del Perú</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Contenido Principal ---
st.markdown("""
<style>
.tooltip {
  position: relative;
  display: inline-block;
  cursor: pointer;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 300px;
  background-color: #f9f9f9;
  color: #333;
  text-align: left;
  border-radius: 6px;
  border: 1px solid #ccc;
  padding: 10px;
  position: absolute;
  z-index: 1;
  top: 100%;
  left: 0;
  font-size: 14px; /* Reducir el tamaño del texto principal */
}

.tooltip:hover .tooltiptext {
  visibility: visible;
}

.tooltip .caption {
  font-size: 10px; /* Hacer el caption más pequeño */
  color: #777; /* Cambiar color si es necesario */
  text-align: center;
  margin-top: 5px;
}
</style>

<div class="tooltip">ℹ️
  <div class="tooltiptext">
      La Municipalidad de Miraflores firmó un convenio con la startup peruana qAIRa para instalar sensores ambientales que permitirán monitorear en tiempo real la calidad del aire en el distrito. El proyecto, financiado por entidades como el Banco Mundial y CONCYTEC, busca proteger la salud pública mediante datos comparados con los estándares ambientales.
      <br><br>
      <img src="https://www.miraflores.gob.pe/wp-content/uploads/2019/10/DSC_0238-1024x681.jpeg" alt="qAIRa" width="100%">
      <div class="caption">Dron de Monitoreo de Calidad de Aire de qAIRa</div>
    </div>
  </div>
""", unsafe_allow_html=True)
st.title("Visualización de Calidad del Aire y Variables Ambientales")
with st.expander("Ver detalles sobre el dataset"):
    
    # Crear tres columnas
    col1, col2, col3 = st.columns(3)

    # Contenido en la primera columna (Identificación y Ubicación)
    with col1:
        st.markdown("""
        🧑‍🔬 **Identificación y Ubicación:**  
        - **Fecha** → Fecha de la medición
        - **Latitud y Longitud** → Coordenadas de la ubicación del dron
        - **Zona** → Estación donde se ubica el dron de monitoreo
        """)
    # Contenido en la segunda columna (Contaminantes del aire)
    with col2:
        st.markdown("""
        🧪 **Contaminantes del aire:**  
        - **CO (ug/m3)** → Monóxido de carbono  
        - **H2S (ug/m3)** → Sulfuro de hidrógeno  
        - **NO2 (ug/m3)** → Dióxido de nitrógeno  
        - **O3 (ug/m3)** → Ozono troposférico  
        - **PM10 (ug/m3)** → Material particulado de hasta 10 micras  
        - **PM2,5 (ug/m3)** → Material particulado fino (hasta 2.5 micras)  
        - **SO2 (ug/m3)** → Dióxido de azufre
        """)

    # Contenido en la tercera columna (Variables meteorológicas)
    with col3:
        st.markdown("""
        🌦️ **Variables meteorológicas:**  
        - **Temperatura (C)** → En grados Celsius  
        - **Humedad (%)** → Relativa  
        - **Presión (Pa)** → Presión atmosférica  
        - **UV** → Índice de radiación ultravioleta  
        - **Ruido (dB)** → Nivel de ruido ambiental
        """)
st.markdown(f"Datos de monitoreo reportados por la Municipalidad de Miraflores. Fuente: [Plataforma Nacional de Datos Abiertos](https://www.datosabiertos.gob.pe/dataset/monitoreo-de-calidad-de-aire-qaira%C2%A0de-la-municipalidad-de-miraflores)")

# Mostrar información de filtros aplicados
filtro_fecha_info = "**todos los datos disponibles**"
if fecha_inicio and fecha_fin:
    filtro_fecha_info = f"datos desde **{fecha_inicio.strftime('%d/%m/%Y')}** hasta **{fecha_fin.strftime('%d/%m/%Y')}**"

filtro_zona_info = ""
if zona_seleccionada:
    filtro_estacion_info = f" para la estación de **{zona_seleccionada}**" if zona_seleccionada != "Todas" else " para **todas** las estaciones"

st.markdown(f"Mostrando {filtro_fecha_info}{filtro_estacion_info}.")


# --- Verificar si hay datos después de filtrar ---
if df_filtrado.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
    st.stop() # Detener si no hay nada que mostrar


# --- Lógica para mostrar la sección seleccionada ---

# 📍 Resumen General
if seleccion == opciones_menu[0]:
    st.header(opciones_menu[0])
    st.markdown("Un vistazo rápido a la ubicación y los indicadores más recientes.")

    col_mapa, col_kpi = st.columns([0.6, 0.4])

    with col_mapa:
        st.subheader("Ubicación de Estación(es)")
        # Usar df_filtrado para las ubicaciones
        locations = df_filtrado[[COL_ZONA, COL_LATITUD, COL_LONGITUD]].dropna().drop_duplicates(subset=[COL_ZONA])

        if not locations.empty:
            locations = locations.rename(columns={COL_LATITUD: 'lat', COL_LONGITUD: 'lon'})
            # st.map(locations[['lat', 'lon']])
            # locations['hover_text'] = ("LATITUD: " + locations['lat'].astype(str) + ", LONGITUD: " + locations['lon'].astype(str))
            locations['size_dummy'] = 10
            locations['Estacion'] = locations[COL_ZONA]
            fig_mapa = px.scatter_map(
                locations,
                lat="lat",
                lon="lon",
                color="Estacion",
                size='size_dummy',
                zoom=10,
                height=500,
                map_style="open-street-map",
            )
            fig_mapa.update_layout(showlegend=False)
            fig_mapa.update_layout(margin=dict(l=0, r=0, t=0, b=50))
            st.plotly_chart(fig_mapa, use_container_width=True)
            nombres_estaciones = locations[COL_ZONA].unique()
            if len(nombres_estaciones) > 0:
                st.write(f"**Estación(es) mostrada(s):** {', '.join(nombres_estaciones)}")
        else:
            st.info("No hay datos de ubicación válidos para los filtros seleccionados.")

    with col_kpi:
        st.subheader("Indicadores Clave Recientes")
        if not df_filtrado.empty and COL_FECHA_HORA in df_filtrado.columns:
            # Ordenar el df_filtrado para obtener el último dato
            df_filtrado_sorted = df_filtrado.sort_values(by=COL_FECHA_HORA, ascending=False)
            latest_data = df_filtrado_sorted.iloc[0]
            fecha_ultimo_dato = latest_data[COL_FECHA_HORA].strftime('%d/%m/%Y %H:%M')
            st.write(f"*Último registro en el periodo/estación:* {fecha_ultimo_dato}")

            cols_metricas = st.columns(2)
            col_idx = 0
            # Usar COLUMNAS_KPI definidas al inicio
            kpis_a_mostrar = [kpi for kpi in COLUMNAS_KPI if kpi in df_filtrado.columns]

            for kpi in kpis_a_mostrar:
                if kpi in latest_data and pd.notna(latest_data[kpi]):
                    valor = latest_data[kpi]
                    unidad = ""
                    if "(" in kpi and ")" in kpi:
                        unidad = kpi[kpi.find("(")+1:kpi.find(")")]
                        nombre_kpi = kpi[:kpi.find("(")].strip()
                    else:
                        nombre_kpi = kpi

                    with cols_metricas[col_idx % 2]:
                        st.metric(label=f"{nombre_kpi} ({unidad})", value=f"{valor:.2f}")
                    col_idx += 1
            
            # Añadir mensaje si no se pudieron mostrar todas las KPIs esperadas
            if col_idx == 0:
                st.info("No hay datos numéricos recientes para mostrar como KPIs.")

        else:
            st.info("No hay datos suficientes para mostrar indicadores recientes.")


# 📈 Tendencias Temporales
elif seleccion == opciones_menu[1]:
    st.header(opciones_menu[1])
    st.markdown("Visualiza la evolución de los contaminantes y otros parámetros en el tiempo.")

    if not COL_FECHA in df_filtrado.columns:
        st.warning("Se requiere la columna 'Fecha' para mostrar tendencias temporales.")
        st.stop()

    # Asegurarse de tener columnas numéricas para seleccionar
    if not columnas_numericas_disponibles:
        st.warning("No hay columnas numéricas disponibles para graficar.")
        st.stop()

    # Selector de variables a graficar
    default_selection_tend = [col for col in ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'O3 (ug/m3)'] if col in columnas_numericas_disponibles]
    variables_seleccionadas = st.multiselect(
        "Selecciona las variables a graficar:",
        options=columnas_numericas_disponibles,
        default=default_selection_tend
    )

    if not variables_seleccionadas:
        st.info("Por favor, selecciona al menos una variable para visualizar la tendencia.")
    elif not df_filtrado.empty:
        # Preparar datos para st.line_chart (necesita índice de fecha)
        df_tendencias = df_filtrado.set_index(COL_FECHA_HORA)
        
        # Seleccionar solo las columnas elegidas
        df_grafico = df_tendencias[variables_seleccionadas+[COL_FECHA,COL_HORA]]

        # Agrupar por fecha, usamos el promedio de los contaminantes
        df_diario = df_grafico.groupby(COL_FECHA)[variables_seleccionadas].mean()
        # Agrupar por hora, usamos el promedio de los contaminantes
        df_por_hora = df_grafico.groupby(COL_HORA)[variables_seleccionadas].mean()

        # Eliminar filas donde *todas* las columnas seleccionadas son NaN para evitar errores
        df_diario = df_diario.dropna(how='all')
        df_por_hora = df_por_hora.dropna(how='all')

        if (not df_diario.empty) and (not df_por_hora.empty):
            st.subheader('Tendencia Diaria')
            st.line_chart(df_diario)
            st.subheader('Tendencia Horaria')
            st.line_chart(df_por_hora)
        else:
            st.warning("No hay datos válidos para las variables seleccionadas en el periodo elegido.")
    else:
        # Esto no debería pasar por el check de df_filtrado.empty al inicio, pero por si acaso
         st.warning("No hay datos filtrados para mostrar tendencias.")

# 📊 Comparativa de Contaminantes
elif seleccion == opciones_menu[2]:
    st.header(opciones_menu[2])
    st.markdown("Compara los niveles promedio de diferentes contaminantes en el periodo seleccionado.")

    # Filtrar solo columnas de contaminantes que estén disponibles y sean numéricas
    contaminantes_disponibles = [col for col in COLUMNAS_CONTAMINANTES if col in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado[col])]

    if not contaminantes_disponibles:
        st.warning("No hay datos de contaminantes numéricos disponibles para comparar.")
        st.stop()
    
    if not df_filtrado.empty:
        # Calcular promedios, ignorando NaN
        promedios = df_filtrado[contaminantes_disponibles].mean(numeric_only=True).dropna()
        df_grouped = df_filtrado[contaminantes_disponibles+[COL_ZONA]].groupby(COL_ZONA).mean(numeric_only=True).dropna()
        if not promedios.empty:
            if zona_seleccionada == "Todas":
                tabs = st.tabs(["Bar Chart", "HeatMap"])
                with tabs[0]:
                    st.bar_chart(promedios)
                    # Mostrar tabla de promedios
                    st.write("Valores promedio:")
                    df_promedios = promedios.reset_index() # 1. Convierte la Serie a DataFrame
                    df_promedios.columns = ['Contaminante', 'Valor Promedio'] # 2. Asigna los nombres deseados
                    st.dataframe(df_promedios) # 3. Muestra el DataFrame con los nombres correctos
                with tabs[1]:                
                    st.dataframe(df_grouped.style.background_gradient(cmap='Blues'))
            else:
                st.bar_chart(promedios)
                # Mostrar tabla de promedios
                st.write("Valores promedio:")
                df_promedios = promedios.reset_index() # 1. Convierte la Serie a DataFrame
                df_promedios.columns = ['Contaminante', 'Valor Promedio'] # 2. Asigna los nombres deseados
                st.dataframe(df_promedios) # 3. Muestra el DataFrame con los nombres correctos
        else:
             st.warning("No se pudieron calcular promedios para los contaminantes en el periodo seleccionado.")

    else:
         st.warning("No hay datos filtrados para calcular promedios.")

# 🌦️ Datos Meteorológicos
elif seleccion == opciones_menu[3]:
    st.header(opciones_menu[3])
    st.markdown("Visualiza las tendencias de los parámetros meteorológicos y el ruido.")

    if not COL_FECHA in df_filtrado.columns:
        st.warning("Se requiere la columna 'Fecha' para mostrar tendencias temporales.")
        st.stop()

    # Filtrar columnas meteorológicas y de ruido disponibles
    meteo_ruido_disponibles = [col for col in (COLUMNAS_METEOROLOGICAS + COLUMNAS_OTRAS) if col in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado[col])]

    if not meteo_ruido_disponibles:
        st.warning("No hay columnas de datos meteorológicos o ruido disponibles para graficar.")
        st.stop()

    default_selection_meteo = [col for col in ['Temperatura (C)', 'Humedad (%)', 'Ruido (dB)'] if col in meteo_ruido_disponibles]
    variables_meteo_seleccionadas = st.multiselect(
        "Selecciona las variables meteorológicas o de ruido a graficar:",
        options=meteo_ruido_disponibles,
        default=default_selection_meteo
    )

    if not variables_meteo_seleccionadas:
        st.info("Por favor, selecciona al menos una variable para visualizar la tendencia.")
    elif not df_filtrado.empty:
        df_tendencias_meteo = df_filtrado.set_index(COL_FECHA)
        df_grafico_meteo = df_tendencias_meteo[variables_meteo_seleccionadas].dropna(how='all')

        if not df_grafico_meteo.empty:
            st.line_chart(df_grafico_meteo)
        else:
            st.warning("No hay datos válidos para las variables seleccionadas en el periodo elegido.")
    else:
        st.warning("No hay datos filtrados para mostrar tendencias.")


# 🔗 Correlaciones
elif seleccion == opciones_menu[4]:
    st.header(opciones_menu[4])
    st.markdown("Explora la relación entre dos variables diferentes.")

    # Usar todas las numéricas disponibles para los selectores
    if len(columnas_numericas_disponibles) < 2:
        st.warning("Se necesitan al menos dos columnas numéricas para explorar correlaciones.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        # Intentar preseleccionar Temperatura si existe
        default_x = 'Temperatura (C)' if 'Temperatura (C)' in columnas_numericas_disponibles else columnas_numericas_disponibles[0]
        variable_x = st.selectbox("Selecciona la variable para el eje X:", options=columnas_numericas_disponibles, index=columnas_numericas_disponibles.index(default_x))
    with col2:
        # Intentar preseleccionar PM2.5 si existe, y que no sea igual a X
        default_y_options = [col for col in ['PM2,5 (ug/m3)', 'O3 (ug/m3)'] if col in columnas_numericas_disponibles and col != variable_x]
        default_y = default_y_options[0] if default_y_options else [col for col in columnas_numericas_disponibles if col != variable_x][0]
        variable_y = st.selectbox("Selecciona la variable para el eje Y:", options=columnas_numericas_disponibles, index=columnas_numericas_disponibles.index(default_y))

    if variable_x == variable_y:
        st.warning("Por favor, selecciona dos variables diferentes para comparar.")
    elif not df_filtrado.empty:
        # Preparar datos para scatter plot, quitando filas donde alguna de las dos variables sea NaN
        df_scatter = df_filtrado[[variable_x, variable_y]].dropna()
        if not df_scatter.empty:
            st.scatter_chart(df_scatter, x=variable_x, y=variable_y)
            
            # Calcular y mostrar correlación de Pearson
            correlation = df_scatter[variable_x].corr(df_scatter[variable_y])
            st.write(f"Correlación de Pearson entre {variable_x} y {variable_y}: **{correlation:.3f}**")
            if abs(correlation) > 0.7:
                st.info("Una correlación fuerte (> 0.7 o < -0.7) sugiere una relación lineal importante.")
            elif abs(correlation) > 0.4:
                st.info("Una correlación moderada (> 0.4 o < -0.4) sugiere alguna relación lineal.")
            else:
                st.info("Una correlación baja (< 0.4 y > -0.4) sugiere una relación lineal débil o inexistente.")
                  
        else:
            st.warning(f"No hay suficientes datos válidos simultáneos para '{variable_x}' y '{variable_y}' en el periodo seleccionado.")

    else:
         st.warning("No hay datos filtrados para mostrar correlaciones.")

# ℹ️ Información Técnica del Dataset
elif seleccion == opciones_menu[5]:
    st.header(opciones_menu[5])
    st.markdown("Detalles sobre los datos cargados.")

    st.subheader("Primeras 5 filas del dataset:")
    st.dataframe(df_filtrado.head())

    st.subheader("Información General y Tipos de Datos:")
    summary_df = pd.DataFrame({
        "Columna": df_filtrado.columns,
        "Tipo": df_filtrado.dtypes.values,
        "Nulos": df_filtrado.isnull().sum().values,
        "No nulos": df_filtrado.notnull().sum().values
    })

    st.write(f"Cantidad de entradas: {df_filtrado.shape[0]}")
    st.dataframe(summary_df)

    st.subheader("Estadísticas Descriptivas (columnas numéricas):")
    st.dataframe(df_filtrado[COLUMNAS_CONTAMINANTES + COLUMNAS_OTRAS + COLUMNAS_METEOROLOGICAS].describe(include='number')) # Incluir solo numéricas explícitamente

    st.subheader("📁 Descargar datos filtrados")
    datos_excel = to_excel(df_filtrado)
    st.download_button(
        label="🗂️ Descargar Excel",
        data=datos_excel,
        file_name="datos_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Pie de Página ---
st.markdown("---")
st.markdown("Dashboard creado con [Streamlit](https://streamlit.io) y [Pandas](https://pandas.pydata.org).")
