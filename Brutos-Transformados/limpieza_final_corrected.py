
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 1. Función para cargar datos con manejo de errores
def cargar_datos(ruta_csv):
    """
    Carga el dataset desde un archivo CSV.
    """
    try:
        df = pd.read_csv(ruta_csv, index_col=0)  # Se elimina la columna 'Unnamed: 0' si existe
        print("✅ Datos cargados correctamente.")
        return df
    except FileNotFoundError:
        print("❌ Error: Archivo no encontrado.")
        return None

# 📌 2. Función para limpiar datos
def limpiar_datos(df):
    """
    Realiza limpieza del dataset:
    - Elimina columnas innecesarias.
    - Traduce nombres de columnas.
    - Optimiza tipos de datos.
    - Maneja valores NaN correctamente.
    """
    if df is None:
        print("❌ Error: El dataset no fue cargado correctamente.")
        return None

    columnas_a_eliminar = ['day_of_week', 'day_name', 'latitud', 'longitud']
    df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

    # Ajuste de nombres de columnas a snake_case
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Diccionario para traducir nombres de columnas
    columnas_traducidas = {
        "job": "ocupacion", "marital": "estado_civil", "education": "educacion",
        "housing": "tiene_hipoteca", "loan": "tiene_prestamo", "contact": "tipo_contacto",
        "duration": "duracion_llamada", "campaign": "contactos_actuales",
        "pdays": "dias_ultimo_contacto", "previous": "contactos_previos",
        "poutcome": "resultado_campana_anterior", "emp_var_rate": "tasa_empleo",
        "cons_price_idx": "indice_precio_consumidor", "cons_conf_idx": "indice_confianza_consumidor",
        "euribor3m": "tasa_interes_3m", "nr_employed": "num_empleados",
        "y": "suscribio", "income": "ingreso", "kidhome": "hijos_ninos",
        "teenhome": "hijos_adolescentes", "dt_customer": "fecha_registro",
        "numwebvisitsmonth": "visitas_web_mes", "pdays_category": "categoria_dias_ultimo_contacto",
        "year_registered": "ano_registro", "month_registered": "mes_registro",
        "contact_date": "fecha_contacto"
    }

    df.rename(columns=columnas_traducidas, inplace=True)

    # Optimizar tipos de datos
    df['fecha_registro'] = pd.to_datetime(df['fecha_registro'], errors='coerce')
    df['fecha_contacto'] = pd.to_datetime(df['fecha_contacto'], errors='coerce')

    # Convertir variables categóricas a category para optimización
    categorias = ["estado_civil", "educacion", "ocupacion", "tipo_contacto", "resultado_campana_anterior"]
    for col in categorias:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Manejo de valores NaN
    df.dropna(subset=['ocupacion', 'estado_civil', 'educacion'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("✅ Datos limpiados correctamente.")
    return df

# 📌 3. Función para detectar y manejar outliers usando el método IQR
def manejar_outliers(df, columnas):
    """
    Elimina outliers usando el método IQR.
    """
    if df is None:
        print("❌ Error: El dataset no fue cargado correctamente.")
        return None

    for col in columnas:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print("✅ Outliers gestionados correctamente.")
    return df

# 📌 4. Función para transformar datos
def transformar_datos(df):
    """
    Realiza transformaciones:
    - Calcula 'dias_desde_registro'.
    - Categoriza la duración de la llamada.
    """
    if df is None:
        print("❌ Error: El dataset no fue cargado correctamente.")
        return None

    df['dias_desde_registro'] = (df['fecha_contacto'] - df['fecha_registro']).dt.days

    # Categorizar la duración de la llamada
    df['categoria_duracion'] = pd.cut(
        df['duracion_llamada'],
        bins=[0, 100, 300, float('inf')],
        labels=['Corta', 'Media', 'Larga'],
        right=False
    )

    print("✅ Transformaciones aplicadas correctamente.")
    return df

# 📌 5. Ejecutar el script
if __name__ == "__main__":
    ruta_csv = "dataset_final.csv"

    df = cargar_datos(ruta_csv)

    if df is not None:
        df = limpiar_datos(df)
        df = manejar_outliers(df, ['duracion_llamada', 'ingreso'])
        df = transformar_datos(df)

        # Guardar dataset optimizado
        df.to_csv("dataset_final_procesado.csv", index=False)
        print("✅ Dataset final guardado como 'dataset_final_procesado.csv'.")
