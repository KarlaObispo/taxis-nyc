import numpy as np
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
VERDE, ROJO, GRIS = "#2f6f62", "#b03a2e", "#7f8c8d"
ZONA_POR_DEFECTO = 161  # Midtown Center


# Acceso a la API
def obtener_api_url():
    url = st.session_state.get("api_url")
    if url:
        return url.rstrip("/")
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return "https://taxis-nyc.onrender.com"

@st.cache_data(ttl=3600, show_spinner=False)
def leer_info(url):
    r = requests.get(f"{url}/info", timeout=90)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def leer_pronostico(url, zona_id, fecha):
    r = requests.get(f"{url}/pronostico", params={"zona_id": zona_id, "fecha": fecha}, timeout=90)
    r.raise_for_status()
    return r.json()


def a_tabla(resp):
    filas = []
    for h in resp["horas"]:
        fila = {"hora": h["hora"], "real": h["real"], "hace_una_semana": h["hace_una_semana"]}
        fila.update(h["cuantiles"])
        filas.append(fila)
    df = pd.DataFrame(filas)
    df["real"] = pd.to_numeric(df["real"])  # None -> NaN
    return df


def etiqueta_fecha(f, ultima):
    d = pd.Timestamp(f)
    texto = f"{DIAS[d.dayofweek]} {d.day}"
    return f"{texto} (mañana)" if f == ultima else texto

@st.dialog("Cómo leer el gráfico")
def ayuda():
    st.markdown(
        "- **Línea verde:** la mediana (p50), el valor más probable.\n"
        "- **Banda:** del cuantil 10% al 90%. Se espera que ~8 de cada 10 horas caigan dentro.\n"
        "- **Puntos rojos:** viajes reales, cuando existen.\n"
        "- **Línea punteada:** lo que pasó a la misma hora hace una semana.\n"
        "- El 1 de febrero no tiene datos reales: es un pronóstico puro."
    )
# Encabezado
col_t, col_b = st.columns([5, 1])
with col_t:
    st.title("Pronóstico de demanda")
    st.caption("Viajes por hora que se esperan en una zona, con su rango de incertidumbre. "
               "El modelo vive en la API.")
with col_b:
    if st.button("Cómo leerlo", icon=":material/help:"):
        ayuda()

#  Conexión 
url = obtener_api_url()
try:
    with st.status("Conectando con la API (la primera vez puede tardar hasta un minuto)...") as estado:
        info = leer_info(url)
        estado.update(label="API conectada", state="complete")
    st.sidebar.badge("API en línea", color="green")
except requests.RequestException:
    st.sidebar.badge("API sin conexión", color="red")
    st.error("No pude conectarme a la API. Revisa la URL en «Configurar API» de la barra lateral "
             "o espera unos segundos si el servicio estaba dormido.")
    st.stop()

# Controles
zonas = sorted(info["zonas"], key=lambda z: z["nombre"])
etiquetas = {z["id"]: f'{z["nombre"]} ({z["distrito"]})' for z in zonas}
ids = list(etiquetas)

# Si el link trae ?zona=...&fecha=..., se usan como valores iniciales
try:
    zona_inicial = int(st.query_params.get("zona", ZONA_POR_DEFECTO))
except ValueError:
    zona_inicial = ZONA_POR_DEFECTO
if zona_inicial not in ids:
    zona_inicial = ZONA_POR_DEFECTO if ZONA_POR_DEFECTO in ids else ids[0]

fechas = info["fechas"]
ultima = fechas[-1]
fecha_inicial = st.query_params.get("fecha", fechas[0])
if fecha_inicial not in fechas:
    fecha_inicial = fechas[0]

zona_id = st.selectbox("Zona", ids, format_func=etiquetas.get,
                       index=ids.index(zona_inicial))
fecha = st.radio("Fecha", fechas, format_func=lambda f: etiqueta_fecha(f, ultima),
                 index=fechas.index(fecha_inicial), horizontal=True)

# Guardar la elección en el link
st.query_params["zona"] = zona_id
st.query_params["fecha"] = fecha

try:
    resp = leer_pronostico(url, zona_id, fecha)
except requests.HTTPError as e:
    if e.response is not None and e.response.status_code == 404:
        st.warning(e.response.json().get("detail", "No hay datos para esa zona o fecha."))
        st.stop()
    st.error("La API respondió con un error. Intenta de nuevo.")
    st.stop()

df = a_tabla(resp)
con_real = df["real"].notna()

#  Tarjetas 
total = df["p50"].sum()
semana = df["hace_una_semana"].sum()
delta = (total / semana - 1) * 100 if semana > 0 else None
pico = df.loc[df["p50"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
with c1.container(border=True):
    st.metric("Viajes esperados en el día", f"{total:,.0f}",
              delta=f"{delta:+.0f}% vs. hace una semana" if delta is not None else None,
              delta_color="off")
with c2.container(border=True):
    st.metric("Hora pico", f"{int(pico['hora']):02d}:00")
    st.caption(f"{pico['p50']:,.0f} viajes")
with c3.container(border=True):
    st.metric("Rango del día (80%)", f"{df['p10'].sum():,.0f} – {df['p90'].sum():,.0f}",
              help="Suma de los cuantiles 10% y 90% de cada hora. Es una aproximación "
                   "conservadora: el rango real del total suele ser algo más estrecho.")
with c4.container(border=True):
    if con_real.any():
        mae = (df.loc[con_real, "real"] - df.loc[con_real, "p50"]).abs().mean()
        mae_sem = (df.loc[con_real, "real"] - df.loc[con_real, "hace_una_semana"]).abs().mean()
        dentro = ((df["real"] >= df["p10"]) & (df["real"] <= df["p90"]))[con_real].mean()
        st.metric("Error promedio por hora", f"{mae:.1f} viajes")
        st.caption(f"Hace una semana: {mae_sem:.1f} · Dentro de la banda: {dentro:.0%} "
                   f"· {df['real'].sum():,.0f} viajes reales")
    else:
        st.metric("Error promedio por hora", "—")
        st.caption("Aún no hay viajes reales para esta fecha")

# Gráfico 
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(df["hora"], df["p10"], df["p90"], color=VERDE, alpha=0.18, label="Rango 10%–90%")
sns.lineplot(data=df, x="hora", y="hace_una_semana", ax=ax, color=GRIS,
             linestyle=":", label="Hace una semana")
sns.lineplot(data=df, x="hora", y="p50", ax=ax, color=VERDE, linewidth=2.5,
             label="Pronóstico (mediana)")
if con_real.any():
    sns.scatterplot(data=df[con_real], x="hora", y="real", ax=ax, color=ROJO, s=45,
                    label="Real", zorder=3)
ax.set(xlabel="Hora del día", ylabel="Viajes", xlim=(0, 23),
       title=f'{resp["zona"]["nombre"]} · {etiqueta_fecha(fecha, ultima)}')
ax.set_xticks(range(0, 24, 2))
ax.legend(loc="upper left", frameon=False, ncol=4)
st.pyplot(fig)
plt.close(fig)

## Tabla y descarga
tabla = df.rename(columns={
    "hora": "Hora", "real": "Real", "hace_una_semana": "Hace una semana",
    "p10": "p10", "p20": "p20", "p30": "p30", "p40": "p40", "p50": "Mediana (p50)",
    "p60": "p60", "p70": "p70", "p80": "p80", "p90": "p90",
})

with st.expander("Ver tabla por hora", expanded=True):
    st.dataframe(
        tabla.round(1),
        hide_index=True,
        width="stretch",
        column_config={
            "Hora": st.column_config.NumberColumn("Hora", format="%d:00"),
            "Real": st.column_config.NumberColumn("Real", format="%.0f"),
            "Mediana (p50)": st.column_config.ProgressColumn(
                "Mediana (p50)", format="%.1f",
                min_value=0, max_value=float(tabla["p90"].max()),
            ),
        },
    )
    st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"),
                       file_name=f"pronostico_{zona_id}_{fecha}.csv", mime="text/csv",
                       key="descargar_csv_pronostico")

# La semana completa
if st.toggle("Ver la semana completa de esta zona"):
    with st.spinner("Cargando la semana..."):
        filas = {etiqueta_fecha(f, ultima): a_tabla(leer_pronostico(url, zona_id, f)).set_index("hora")["p50"]
                 for f in fechas}
    mapa = pd.DataFrame(filas).T
    fig2, ax2 = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(mapa, cmap="YlGnBu", ax=ax2, cbar_kws={"label": "Viajes (mediana)"})
    ax2.set(xlabel="Hora del día", ylabel="", title="Demanda esperada por día y hora")
    st.pyplot(fig2)
    plt.close(fig2)