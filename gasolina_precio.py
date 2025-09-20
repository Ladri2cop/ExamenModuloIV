import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --- Configuración de página ---
st.set_page_config(
    page_title="Predicción Gasolina",
    page_icon="⛽",
    layout="centered"
)

st.title("Predicción del precio de gasolina en México")
st.caption("Dataset: **gasolina_precios.csv** | Variables: estado, año, mes → precio")


# --- Lectura de datos con cache ---
try:
    use_cache = st.cache_data
except AttributeError:
    use_cache = st.cache


DATA_FILE = "gasolina_precios.csv"

@use_cache
def cargar_datos():
    """Lee el CSV probando diferentes formatos y valida las columnas requeridas."""
    opciones = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
    ]
    ultimo_error, data = None, None
    for cfg in opciones:
        try:
            temp = pd.read_csv(DATA_FILE, **cfg)
            temp.columns = [c.strip().lower() for c in temp.columns]
            if "anio" in temp.columns and "año" not in temp.columns:
                temp = temp.rename(columns={"anio": "año"})
            if {"estado", "año", "mes", "precio"}.issubset(temp.columns):
                data = temp[["estado", "año", "mes", "precio"]].copy()
                break
        except Exception as e:
            ultimo_error = e
    if data is None:
        raise RuntimeError(f"No fue posible leer el archivo. Último error: {ultimo_error}")

    # Conversión de tipos
    data["estado"] = data["estado"].astype(str)
    for col in ["año", "mes", "precio"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna().reset_index(drop=True)


df = cargar_datos()

with st.expander("📊 Vista previa del dataset", expanded=False):
    st.dataframe(df.head(15))


# --- Preparación de variables ---
X = df[["estado", "año", "mes"]]
y = df["precio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- Modelo ---
transformador = ColumnTransformer(
    [("estado_ohe", OneHotEncoder(handle_unknown="ignore"), ["estado"])],
    remainder="passthrough"
)

modelo = Pipeline([
    ("prep", transformador),
    ("linreg", LinearRegression())
])

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)


# --- Métricas ---
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Rendimiento del modelo")
c1, c2 = st.columns(2)
c1.metric("RMSE", f"{rmse:,.4f}")
c2.metric("R²", f"{r2:,.4f}")


# --- Predicción interactiva ---
st.header("Probar una predicción")

col1, col2 = st.columns(2)
estados = sorted(df["estado"].unique().tolist())

with col1:
    estado_sel = st.selectbox("Estado", estados, index=0)
with col2:
    anio_sel = st.number_input(
        "Año",
        min_value=int(df["año"].min()),
        max_value=int(df["año"].max()),
        value=int(df["año"].max()),
        step=1
    )

mes_sel = st.slider("Mes", 1, 12, int(df["mes"].median()))

entrada = pd.DataFrame([{
    "estado": estado_sel,
    "año": int(anio_sel),
    "mes": int(mes_sel)
}])

if st.button("Predecir precio"):
    prediccion = modelo.predict(entrada)[0]
    st.success(f"Precio estimado: **{prediccion:,.4f}**")




