import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def cargar():
    zonas = pd.read_csv(DATA / "zonas_resumen.csv").set_index("zona_id")
    zonas = zonas.rename(columns={"pct_22_5": "pct_noche"})
    perfil = pd.read_csv(DATA / "perfil_horario.csv").set_index("zona_id")
    perfil.columns = range(24)
    with open(DATA / "silueta.json") as f:
        silueta = json.load(f)
    return zonas, perfil, silueta


zonas, perfil, silueta = cargar()

st.title("Segmentación de zonas")
st.caption("K-Means agrupa las zonas por cómo se usan los taxis. PCA las resume en dos ejes para poder dibujarlas.")

# --- Filtro de grupos ---
nombres = zonas.groupby("grupo")["nombre_grupo"].first().to_dict()
seleccion = st.pills("Grupos", list(nombres.values()), selection_mode="multi", default=list(nombres.values()))
ids = [g for g, n in nombres.items() if n in seleccion]
if not ids:
    st.info("Selecciona al menos un grupo.")
    st.stop()
filtradas = zonas[zonas["grupo"].isin(ids)]

# --- Tarjetas por grupo ---
cols = st.columns(len(nombres))
for col, (g, nombre) in zip(cols, nombres.items()):
    d = zonas[zonas["grupo"] == g]
    with col, st.container(border=True):
        st.markdown(f"**{nombre}**")
        st.metric("Zonas", len(d))
        st.caption(
            f"{int(d['viajes_dia'].sum()):,} viajes al día · "
            f"{d['distancia_mediana'].median():.1f} millas · {d['pct_noche'].mean():.0%} de noche"
        )

# --- Dispersión PCA + perfil horario ---
izq, der = st.columns([1.2, 1])
puntos = filtradas.reset_index()
fig = px.scatter(
    puntos, x="pc1", y="pc2", color="nombre_grupo", size="viajes_dia",
    custom_data=["zona_id"], hover_name="nombre",
    labels={"pc1": "Componente 1", "pc2": "Componente 2", "nombre_grupo": "Grupo"},
)
evento = izq.plotly_chart(fig, on_select="rerun", selection_mode=("points", "box", "lasso"))
izq.caption("Selecciona zonas en el gráfico (clic, caja o lazo) para compararlas.")

elegidas = [p["customdata"][0] for p in evento.selection.points]
if not elegidas:
    elegidas = filtradas.nlargest(3, "viajes_dia").index.tolist()

df_h = pd.DataFrame({zonas.loc[z, "nombre"]: perfil.loc[z].values for z in elegidas})
df_h.index.name = "Hora del día"
der.markdown("**Perfil horario de las zonas**")
der.line_chart(df_h)

# --- Tabla de detalle ---
st.subheader("Detalle por zona")
tabla = filtradas.sort_values("viajes_dia", ascending=False).head(10).copy()
tabla["pct_noche"] = tabla["pct_noche"] * 100
tabla["perfil_24h"] = [perfil.loc[z].round(4).tolist() for z in tabla.index]

st.dataframe(
    tabla[["nombre", "distrito", "nombre_grupo", "viajes_dia", "distancia_mediana",
           "pct_noche", "propina_mediana", "perfil_24h"]],
    column_config={
        "nombre": "Zona",
        "distrito": "Distrito",
        "nombre_grupo": "Grupo",
        "viajes_dia": "Viajes al día",
        "distancia_mediana": "Distancia mediana",
        "pct_noche": st.column_config.ProgressColumn("Viajes de noche", format="%.1f%%", min_value=0, max_value=100),
        "propina_mediana": st.column_config.NumberColumn("Propina mediana", format="$%.2f"),
        "perfil_24h": st.column_config.LineChartColumn("Perfil de 0 a 23 h"),
    },
    hide_index=True,
    use_container_width=True,
)

# --- Justificar k ---
with st.expander(f"¿Por qué {len(nombres)} grupos?"):
    sil = pd.Series(silueta)
    sil.index = sil.index.astype(int)
    st.line_chart(sil)
    st.caption("Silueta para k = 2 a 7: valores más altos indican grupos mejor separados.")