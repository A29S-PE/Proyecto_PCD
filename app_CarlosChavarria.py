import streamlit as st
import pandas as pd
import os
from io import StringIO

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Calidad del Aire Miraflores - Lima, Perú",
    page_icon="🌬️",
    layout="wide"
)

# --- Constantes ---
NOMBRE_ARCHIVO = 'Monitoreo.xlsx'
# Columnas a limpiar (convertir a numérico)
COLUMNAS_A_LIMPIAR = ['H2S (ug/m3)', 'SO2 (ug/m3)']
# Columnas de coordenadas
COL_LATITUD = 'Latitud'
COL_LONGITUD = 'Longitud'
COL_FECHA = 'Fecha'
COL_ESTACION = 'NOMBRE DE LA UO'

# Definir listas de columnas para facilitar selección
# (Excluye códigos, nombres, fechas, coordenadas de las listas de variables medibles)
COLUMNAS_CONTAMINANTES = ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)']
COLUMNAS_METEOROLOGICAS = ['Humedad (%)', 'Presion (Pa)', 'Temperatura (C)', 'UV']
COLUMNAS_OTRAS = ['Ruido (dB)']

# Columnas clave para KPIs
# Usemos las listas definidas arriba para mantener consistencia
COLUMNAS_KPI = COLUMNAS_CONTAMINANTES + COLUMNAS_OTRAS

# --- Funciones ---

@st.cache_data
def load_data(file_path):
    """Carga los datos desde el archivo Excel y realiza limpieza básica."""
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la misma carpeta que app.py.")
        st.stop()

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()

        if COL_FECHA in df.columns:
            df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')
            # Ordenar por fecha una vez cargados los datos
            df = df.sort_values(by=COL_FECHA)
        else:
            st.warning(f"Advertencia: No se encontró la columna '{COL_FECHA}'. Las funciones de tiempo no funcionarán correctamente.")

        for col in COLUMNAS_A_LIMPIAR:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Advertencia: No se encontró la columna '{col}' para limpieza.")

        if COL_LATITUD not in df.columns or COL_LONGITUD not in df.columns:
             st.warning(f"Advertencia: No se encontraron las columnas '{COL_LATITUD}' y/o '{COL_LONGITUD}'. El mapa no funcionará.")

        # Convertir todas las columnas numéricas potenciales a float para consistencia en gráficos
        columnas_numericas_potenciales = COLUMNAS_CONTAMINANTES + COLUMNAS_METEOROLOGICAS + COLUMNAS_OTRAS
        for col in columnas_numericas_potenciales:
             if col in df.columns:
                  # Forzar conversión a numérico, por si alguna quedó como object tras la carga inicial
                  df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        st.error(f"Ocurrió un error al cargar o procesar el archivo Excel: {e}")
        st.info("Asegúrate de que el archivo no esté corrupto y sea un .xlsx válido.")
        st.info("Puede que necesites instalar 'openpyxl': pip install openpyxl")
        st.stop()

# --- Carga de Datos ---
df_original = load_data(NOMBRE_ARCHIVO)

# Crear lista de columnas numéricas disponibles DESPUÉS de la limpieza
# Estas son las columnas que se podrán seleccionar en los gráficos
columnas_numericas_disponibles = [
    col for col in (COLUMNAS_CONTAMINANTES + COLUMNAS_METEOROLOGICAS + COLUMNAS_OTRAS)
    if col in df_original.columns and pd.api.types.is_numeric_dtype(df_original[col])
]

# Hacemos una copia para los filtros
df = df_original.copy()

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Navegación")

# --- Filtro de Fecha (solo si la columna Fecha existe y tiene datos) ---
fecha_inicio = None
fecha_fin = None
df_filtrado = df.copy() # Inicializar df_filtrado

if COL_FECHA in df.columns and not df[COL_FECHA].isnull().all():
    df_valid_dates = df.dropna(subset=[COL_FECHA]).copy() # Usar df con fechas válidas para min/max

    if not df_valid_dates.empty:
        # Ya está ordenado en load_data
        fecha_min = df_valid_dates[COL_FECHA].min().date()
        fecha_max = df_valid_dates[COL_FECHA].max().date()

        #st.sidebar.header("Filtrar por Fecha")
        # Asegurarse de que los valores por defecto estén dentro del rango min/max
        default_start = max(fecha_min, fecha_min)
        default_end = min(fecha_max, fecha_max)

        fecha_inicio = st.sidebar.date_input("Fecha de inicio", default_start, min_value=fecha_min, max_value=fecha_max)
        fecha_fin = st.sidebar.date_input("Fecha de fin", default_end, min_value=fecha_min, max_value=fecha_max)

        if fecha_inicio > fecha_fin:
            st.sidebar.error("Error: La fecha de inicio no puede ser posterior a la fecha de fin.")
            st.stop()
        else:
            # Aplicar filtro de fecha
            fecha_inicio_dt = pd.to_datetime(fecha_inicio)
            fecha_fin_dt = pd.to_datetime(fecha_fin) + pd.Timedelta(days=1) # Incluir todo el día final
            
            # Filtrar usando el df original limpio (df_valid_dates ya no tiene NaT)
            df_filtrado = df_valid_dates[(df_valid_dates[COL_FECHA] >= fecha_inicio_dt) & (df_valid_dates[COL_FECHA] < fecha_fin_dt)].copy()

    else:
         st.sidebar.warning("No hay fechas válidas en los datos para filtrar.")
         # df_filtrado sigue siendo una copia completa pero sin filas con NaT en Fecha
         df_filtrado = df_valid_dates.copy()

else:
    st.sidebar.info("La columna de Fecha no está disponible o está vacía. No se pueden aplicar filtros de fecha.")
    # df_filtrado sigue siendo una copia completa del df original cargado

# --- Selector de Estación ---
estaciones_disponibles = df_filtrado[COL_ESTACION].unique() if COL_ESTACION in df_filtrado.columns else []
estacion_seleccionada = None

if len(estaciones_disponibles) > 1:
    st.sidebar.header("Filtrar por Estación")
    # Opción para ver todas las estaciones (promedio o juntas, dependiendo del gráfico)
    opciones_estacion = ["Todas"] + list(estaciones_disponibles)
    estacion_seleccionada = st.sidebar.selectbox("Selecciona una estación:", opciones_estacion)
    
    if estacion_seleccionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado[COL_ESTACION] == estacion_seleccionada].copy()
    # Si es "Todas", df_filtrado ya contiene todas las estaciones del rango de fecha

# --- Menú Principal ---
opciones_menu = [
    "📍 Resumen General",
    "📈 Tendencias Temporales",
    "📊 Comparativa de Contaminantes",
    "🌦️ Datos Meteorológicos",
    "🔗 Correlaciones",
    "ℹ️ Información del Dataset"
]
# Asegurarnos de que el índice 0 siempre sea Resumen General
seleccion = st.sidebar.radio("Selecciona una sección:", opciones_menu, index=0)

# --- Contenido Principal ---
st.title("🌬️ Dashboard de Calidad del Aire - Miraflores")
st.markdown(f"Datos de monitoreo reportados por la Municipalidad de Miraflores. Fuente: [Plataforma Nacional de Datos Abiertos](https://www.datosabiertos.gob.pe/search/type/dataset)")

# Mostrar información de filtros aplicados
filtro_fecha_info = "Todos los datos disponibles"
if fecha_inicio and fecha_fin:
    filtro_fecha_info = f"Desde **{fecha_inicio.strftime('%d/%m/%Y')}** hasta **{fecha_fin.strftime('%d/%m/%Y')}**"

filtro_estacion_info = ""
if estacion_seleccionada:
    filtro_estacion_info = f" para la estación **{estacion_seleccionada}**" if estacion_seleccionada != "Todas" else " para **todas** las estaciones"

st.markdown(f"Mostrando datos {filtro_fecha_info}{filtro_estacion_info}.")

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
        if COL_LATITUD in df_filtrado.columns and COL_LONGITUD in df_filtrado.columns and not df_filtrado[[COL_LATITUD, COL_LONGITUD]].isnull().all().all():
            # Usar df_filtrado para las ubicaciones
            locations = df_filtrado[[COL_ESTACION, COL_LATITUD, COL_LONGITUD]].dropna().drop_duplicates(subset=[COL_ESTACION])

            if not locations.empty:
                locations = locations.rename(columns={COL_LATITUD: 'lat', COL_LONGITUD: 'lon'})
                st.map(locations[['lat', 'lon']])
                nombres_estaciones = locations[COL_ESTACION].unique()
                if len(nombres_estaciones) > 0:
                    st.write(f"**Estación(es) mostrada(s):** {', '.join(nombres_estaciones)}")
            else:
                st.info("No hay datos de ubicación válidos para los filtros seleccionados.")
        else:
            st.warning("No se pueden mostrar las ubicaciones. Faltan datos de Latitud/Longitud.")

    with col_kpi:
        st.subheader("Indicadores Clave Recientes")
        if not df_filtrado.empty and COL_FECHA in df_filtrado.columns:
            # Ordenar el df_filtrado para obtener el último dato
            df_filtrado_sorted = df_filtrado.sort_values(by=COL_FECHA, ascending=False)
            latest_data = df_filtrado_sorted.iloc[0]
            fecha_ultimo_dato = latest_data[COL_FECHA].strftime('%d/%m/%Y %H:%M')
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
        df_tendencias = df_filtrado.set_index(COL_FECHA)
        
        # Seleccionar solo las columnas elegidas
        df_grafico = df_tendencias[variables_seleccionadas]

        # Eliminar filas donde *todas* las columnas seleccionadas son NaN para evitar errores
        df_grafico = df_grafico.dropna(how='all')

        if not df_grafico.empty:
            st.line_chart(df_grafico)
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

        if not promedios.empty:
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
        default_y_options = [col for col in ['PM2.5 (ug/m3)', 'O3 (ug/m3)'] if col in columnas_numericas_disponibles and col != variable_x]
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

# ℹ️ Información del Dataset => VER SI SE QUEDA, PUES SON DATOS QUE PODRÍAN CONFUNDIR AL PÚBLICO GENERAL
elif seleccion == opciones_menu[5]:
    st.header(opciones_menu[5])
    st.markdown("Detalles sobre los datos cargados (antes de aplicar filtros).")

    st.subheader("Primeras 5 filas del dataset original:")
    st.dataframe(df_original.head())

    st.subheader("Información General y Tipos de Datos:")
    buffer = StringIO()
    df_original.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Estadísticas Descriptivas (columnas numéricas):")
    st.dataframe(df_original.describe(include='number')) # Incluir solo numéricas explícitamente

# --- Pie de Página ---
st.markdown("---")
st.markdown("Dashboard creado con [Streamlit](https://streamlit.io) y [Pandas](https://pandas.pydata.org).")