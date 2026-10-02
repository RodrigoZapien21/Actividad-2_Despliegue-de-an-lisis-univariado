"""
=====================================================================================
Actividad 2 | Despliegue de Análisis Univariado · Socio formador - CLV
Etapa I · Modelado explicativo · Extracción de características
=====================================================================================
"""

# ------------------------------------------------------------------ 1. IMPORTACIONES
import re                      # expresiones regulares (normalizar nombres)
import unicodedata             # quitar acentos
from pathlib import Path       # rutas de archivos

import numpy as np             # cálculo numérico
import pandas as pd            # tablas (DataFrames)
import plotly.express as px    # gráficos (barras, dona, heatmap)
import streamlit as st         # servidor web en localhost

# ------------------------------------------------------------------ 2. CONFIGURACIÓN
st.set_page_config(page_title="CLV | Análisis Univariado", page_icon="📊", layout="wide")
DATA_DIR = Path(__file__).parent          # los CSV están junto a este script
PALETA = px.colors.qualitative.Bold       # colores de la dona
LOGO = DATA_DIR / "clv_logo.jpg" 

# ------------------------------------------------------------------ 3. HOJAS Y LAS 15 VARIABLES MÁS RELEVANTES
# keyword -> palabra que identifica el CSV por su nombre (ej. historico_final.csv -> "historico")
# vars    -> variables categóricas elegidas (3 a 4 por hoja cuando es posible; suman 15)
# color   -> color distintivo de la hoja
HOJAS = {
    "Ventas":        {"icon": "🛒", "keyword": "ventas",        "color": "#6C5CE7", "vars": ["marca", "estado", "vendedor"]},
    "Existencias":   {"icon": "📦", "keyword": "existencias",   "color": "#00B894", "vars": ["marca"]},
    "Histórico":     {"icon": "🧾", "keyword": "historico",     "color": "#0984E3", "vars": ["nombre_articulo"]},
    "Clasificación": {"icon": "🗂️", "keyword": "clasificacion", "color": "#E17055", "vars": ["categoria"]},
    "Salidas":       {"icon": "🚚", "keyword": "salidas",       "color": "#C98F00", "vars": ["marca"]},
    "Negadas":       {"icon": "🚫", "keyword": "negadas",       "color": "#D63031", "vars": ["nombre_etapa", "estatus", "motivo_cierre_perdido"]},
    "Negocios":      {"icon": "🤝", "keyword": "negocios",      "color": "#00A5A0", "vars": ["tipo_negocio", "nombre_pipeline", "nombre_zona", "vendedor"]},
    "Metas":         {"icon": "🎯", "keyword": "metas",         "color": "#E84393", "vars": ["estado"]},
}
# Orden de búsqueda de archivos: "negadas" antes que "ventas" para no confundirlos
ORDEN_BUSQUEDA = ["Negadas", "Histórico", "Clasificación", "Existencias", "Salidas", "Negocios", "Metas", "Ventas"]

# ------------------------------------------------------------------ 4. ESTILO (CSS)
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);padding:1.2rem 1.6rem;border-radius:16px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.8rem;color:#fff}.hero p{margin:.2rem 0 0;opacity:.8}
.kpi{border-radius:14px;padding:.8rem 1rem;text-align:center;color:#fff;box-shadow:0 4px 12px rgba(0,0,0,.15)}
.kpi h3{margin:0;font-size:1.6rem;color:#fff}.kpi span{font-size:.85rem;opacity:.95}
.banda{border-radius:12px;padding:.6rem 1.1rem;color:#fff;margin:1.4rem 0 .6rem;font-size:1.15rem;font-weight:600}
.banda small{opacity:.85;font-weight:400;margin-left:.6rem}
</style>""", unsafe_allow_html=True)


def kpi(col, icon, valor, etiqueta, color):
    """Tarjeta KPI de color con ícono."""
    col.markdown(f'<div class="kpi" style="background:{color}"><span>{icon} {etiqueta}</span><h3>{valor}</h3></div>', unsafe_allow_html=True)


def banda(texto, sub, color):
    """Encabezado de color para separar secciones."""
    st.markdown(f'<div class="banda" style="background:{color}">{texto}<small>{sub}</small></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ 5. CARGA DE DATOS
def norm(s):
    """Minúsculas, sin acentos, con '_' en vez de espacios/símbolos."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def leer_csv(path):
    """Lee un CSV probando codificaciones y detectando el separador automáticamente."""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner="Cargando archivos…")
def cargar():
    """Asigna cada CSV de la carpeta a su hoja según el nombre del archivo. Devuelve {hoja: DataFrame}."""
    datos = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        k = norm(f.stem)
        for hoja in ORDEN_BUSQUEDA:
            if hoja not in datos and HOJAS[hoja]["keyword"] in k:
                df = leer_csv(f)
                if df is not None:
                    df.columns = [norm(c) for c in df.columns]        # nombres de columna homogéneos
                    for c in df.columns:
                        if "fecha" in c:                              # fechas -> datetime (para el heatmap por mes)
                            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                    datos[hoja] = df
                break
    return datos


# ------------------------------------------------------------------ 6. FUNCIONES AUXILIARES
def frecuencias(s, top_n):
    """Tabla de frecuencias; lo que queda fuera del Top N se agrupa en 'OTROS'."""
    vc = s.astype(str).value_counts()
    t = pd.DataFrame({"categoría": vc.index, "frecuencia": vc.values})
    if len(t) > top_n:
        resto = t.iloc[top_n:]["frecuencia"].sum()
        t = pd.concat([t.head(top_n), pd.DataFrame({"categoría": ["OTROS"], "frecuencia": [resto]})], ignore_index=True)
    t["%"] = (t["frecuencia"] / t["frecuencia"].sum() * 100).round(2)
    return t


def cramers_v(a, b):
    """Asociación (0 a 1) entre dos variables categóricas."""
    ct = pd.crosstab(a, b).values.astype(float)
    n = ct.sum()
    esperado = np.outer(ct.sum(1), ct.sum(0)) / n
    r, c = ct.shape
    return float(np.sqrt((((ct - esperado) ** 2) / esperado).sum() / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


# ------------------------------------------------------------------ 7. SIDEBAR (única parte interactiva)
DATOS = cargar()
st.sidebar.title("📊 CLV")
st.sidebar.caption("Elige las variables categóricas que quieres visualizar")

# Lista de las 15 variables: (hoja, variable)
TODAS = [(h, v) for h, cfg in HOJAS.items() for v in cfg["vars"]]


def etiqueta(hv):
    """Texto con ícono que aparece en el selector del sidebar."""
    return f"{HOJAS[hv[0]]['icon']} {hv[0]} · {hv[1]}"


elegidas = st.sidebar.multiselect("🏷️ Variables", TODAS, default=TODAS, format_func=etiqueta)
top_n = st.sidebar.slider("🔝 Top N categorías", 2, 15, 8)
st.sidebar.info(f"Variables seleccionadas: {len(elegidas)} de {len(TODAS)}")

# ------------------------------------------------------------------ 8. ENCABEZADO Y KPIs
# Encabezado: título a la izquierda y logo en la esquina superior derecha
col_hero, col_logo = st.columns([5, 1])
with col_hero:
    st.markdown('<div class="hero"><h1>📊 CLV · Análisis Univariado de Variables Categóricas</h1>'
                '<p>Etapa I · Modelado explicativo · Extracción de características</p></div>', unsafe_allow_html=True)
with col_logo:
    if LOGO.exists():
        st.image(str(LOGO), use_container_width=True)

if not DATOS:
    st.error("No encontré CSV. Colócalos en la misma carpeta que Actividad2.py (Actividad2_Despliegue).")
    st.stop()

# Solo se muestran las variables elegidas cuyo archivo y columna existen
validas = [(h, v) for h, v in elegidas if h in DATOS and v in DATOS[h].columns]
faltan = sorted({h for h, _ in elegidas if h not in DATOS})
if faltan:
    st.warning("No encontré el CSV de: " + ", ".join(faltan))
if not validas:
    st.info("Selecciona al menos una variable en el panel izquierdo.")
    st.stop()

hojas_usadas = sorted({h for h, _ in validas})
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "🏷️", len(validas), "Variables", "#6C5CE7")
kpi(k2, "📑", len(hojas_usadas), "Hojas", "#00B894")
kpi(k3, "🧾", f"{sum(len(DATOS[h]) for h in hojas_usadas):,}", "Registros", "#E17055")
kpi(k4, "🔝", top_n, "Top N categorías", "#0984E3")

# ------------------------------------------------------------------ 9. ANÁLISIS UNIVARIADO (todo se despliega automáticamente)
# Por cada variable: barras + dona + tabla de frecuencias (máximo 3 elementos por variable)
for hoja, var in validas:
    cfg = HOJAS[hoja]
    df = DATOS[hoja]
    T = frecuencias(df[var], top_n)

    banda(f"{cfg['icon']} {hoja} · {var}",
          f"{df[var].nunique():,} categorías · moda: {T.iloc[0, 0][:25]} ({T.iloc[0, 2]}%)", cfg["color"])

    c1, c2, c3 = st.columns([1.3, 1, 0.9])
    # Gráfico de barras (color de la hoja)
    f = px.bar(T, x="categoría", y="frecuencia", text="%", color_discrete_sequence=[cfg["color"]])
    f.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title=None)
    c1.plotly_chart(f, use_container_width=True)
    # Dona
    f = px.pie(T, names="categoría", values="frecuencia", hole=.5, color_discrete_sequence=PALETA)
    f.update_layout(height=340, margin=dict(t=20, b=10), showlegend=False)
    f.update_traces(textposition="inside", textinfo="percent")
    c2.plotly_chart(f, use_container_width=True)
    # Tabla con degradado en el porcentaje
    c3.dataframe(T.style.background_gradient(subset=["%"], cmap="YlGnBu").format({"%": "{:.2f}"}),
                 use_container_width=True, height=340, hide_index=True)

# ------------------------------------------------------------------ 10. MAPAS DE CALOR
banda("🔥 Mapas de calor", "Asociación entre variables y comportamiento mensual", "#2c5364")

# 10a) Asociación (V de Cramér) entre las variables elegidas de una misma hoja
st.subheader("🧩 Asociación entre variables de la misma hoja (V de Cramér)")
hubo = False
cols = st.columns(2)
i = 0
for hoja in hojas_usadas:
    vs = [v for h, v in validas if h == hoja]
    if len(vs) > 1:
        hubo = True
        muestra = DATOS[hoja][vs].sample(min(len(DATOS[hoja]), 20000), random_state=1)
        M = pd.DataFrame([[cramers_v(muestra[a], muestra[b]) for b in vs] for a in vs], index=vs, columns=vs)
        f = px.imshow(M, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Magma", title=f"{HOJAS[hoja]['icon']} {hoja}")
        f.update_layout(height=340)
        cols[i % 2].plotly_chart(f, use_container_width=True)
        i += 1
if not hubo:
    st.caption("Selecciona 2 o más variables de una misma hoja (Ventas, Negadas o Negocios) para ver este mapa.")

# 10b) Variable × mes (solo hojas con una columna de fecha)
st.subheader("📅 Frecuencia por mes (variable × mes)")
cols = st.columns(2)
i = 0
for hoja, var in validas:
    df = DATOS[hoja]
    fechas = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not fechas:
        continue
    top = df[var].value_counts().head(min(top_n, 10)).index                     # Top categorías (máx. 10 para legibilidad)
    sub = df[df[var].isin(top)].dropna(subset=[fechas[0]])
    ct = pd.crosstab(sub[var].astype(str), sub[fechas[0]].dt.to_period("M").astype(str))
    f = px.imshow(ct, aspect="auto", color_continuous_scale="YlOrRd", title=f"{HOJAS[hoja]['icon']} {hoja} · {var}")
    f.update_layout(height=340, xaxis_title=None, yaxis_title=None)
    cols[i % 2].plotly_chart(f, use_container_width=True)
    i += 1