import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

DATOS = Path(__file__).parent.parent / "data"
VERDE, GRIS, NARANJA = "#2f6f62", "#7f8c8d", "#d9822b"


@st.cache_data
def cargar():
    with open(DATOS / "ficha_modelo.json", encoding="utf-8") as f:
        ficha = json.load(f)
    grupos = pd.read_csv(DATOS / "ficha_por_grupo.csv")
    return ficha, grupos


ficha, grupos = cargar()
mae = ficha["mae"]

st.title("Ficha del modelo")
st.caption("Qué hace el modelo, qué tan bien lo hace y en qué casos no conviene confiar en él.")

# Para qué sirve 
with st.container(border=True):
    st.subheader(":material/target: Para qué sirve")
    st.markdown(
        f"Pronostica cuántos viajes en taxi empezarán **cada hora** en "
        f"**{ficha['n_zonas']} zonas** de Nueva York, con un rango de incertidumbre "
        f"(cuantiles del 10% al 90%). Sirve para **planificar la flota** con un día de anticipación."
    )
    st.markdown(
        f"**Datos:** viajes de taxis amarillos de NYC TLC, enero de 2024. "
        f"Entrenamiento del {ficha['entrenamiento']['desde'][:10]} al {ficha['entrenamiento']['hasta'][:10]}; "
        f"prueba del {ficha['prueba']['desde'][:10]} al {ficha['prueba']['hasta'][:10]} "
        f"({ficha['n_filas_prueba']:,} zonas-hora evaluadas)."
    )

#  Qué tan bien funciona 
st.subheader(":material/query_stats: Qué tan bien funciona")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.metric("Error del modelo", f"{mae['modelo']:.1f} viajes/hora",
              delta=f"{ficha['mejora_vs_semana_pct']:.1f}% mejor que «igual que hace una semana»",
              help="Error absoluto medio (MAE) de la mediana p50 en la semana de prueba.")
with c2.container(border=True):
    st.metric("Cobertura del rango 10%–90%", f"{ficha['cobertura_80']:.1%}",
              help="Idealmente cerca de 80%: de cada 10 horas, unas 8 deben caer dentro del rango.")
with c3.container(border=True):
    st.metric("Zonas evaluadas", f"{ficha['n_zonas']}")

st.markdown(
    f"El modelo le gana a la regla de «igual que hace una semana» por muy poco -->MAE 11.60 (Es decir 0.28 menor) "
    f"({mae['modelo']:.1f} contra {mae['hace_una_semana']:.1f} viajes por hora). "
    f"Su principal ventaje es frente a «igual que ayer» ({mae['ayer']:.1f}). "
    f"El rango cubre {ficha['cobertura_80']:.1%} de los casos, algo por encima del 80% buscado: "
    f"es ligeramente conservador."
)

# Gráficos (seaborn)
grupos["mejora_vs_semana_pct"] = (1 - grupos["mae_modelo"] / grupos["mae_semana"]) * 100
sns.set_theme(style="whitegrid")
g1, g2 = st.columns(2)

with g1:
    reglas = pd.DataFrame({
        "regla": ["Modelo (p50)", "Igual que hace una semana", "Igual que ayer"],
        "mae": [mae["modelo"], mae["hace_una_semana"], mae["ayer"]],
    })
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(data=reglas, y="regla", x="mae", ax=ax, palette=[VERDE, GRIS, GRIS], hue="regla", legend=False)
    ax.set(title="Error frente a reglas simples", xlabel="Viajes por hora (error medio)", ylabel="")
    st.pyplot(fig)
    plt.close(fig)

with g2:
    orden = grupos.sort_values("error_relativo_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(data=orden, y="nombre_grupo", x="error_relativo_pct", ax=ax, color=NARANJA)
    ax.set(title="Error según el tipo de zona", xlabel="Error como % de la demanda media", ylabel="")
    st.pyplot(fig)
    plt.close(fig)

peor = grupos.loc[grupos["error_relativo_pct"].idxmax()]
mejor = grupos.loc[grupos["mejora_vs_semana_pct"].idxmax()]
st.caption(
    f"El error porcentual es más alto donde la demanda es baja («{peor['nombre_grupo']}», "
    f"{peor['error_relativo_pct']:.0f}%) porque con pocos viajes por hora el azar pesa más: "
    f"es una propiedad de los conteos pequeños. Frente a la regla «igual que hace una semana», "
    f"la mayor mejora está en «{mejor['nombre_grupo']}» ({mejor['mejora_vs_semana_pct']:.0f}%)."
)

# Cobertura por zona 
@st.cache_data
def cargar_zonas():
    return pd.read_csv(DATOS / "ficha_por_zona.csv")


por_zona = cargar_zonas()

st.subheader(":material/table_view: Cobertura por zona")
st.caption("Cada zona se evalúa con una sola semana (168 horas), así que su cobertura individual "
           "es más variable que la global.")

opciones = ["Todos"] + sorted(por_zona["nombre_grupo"].dropna().unique())
filtro = st.selectbox("Tipo de zona", opciones)
vista = por_zona if filtro == "Todos" else por_zona[por_zona["nombre_grupo"] == filtro]
vista = vista.sort_values("cobertura").copy()
vista["cobertura_pct"] = vista["cobertura"] * 100

columnas = ["zona", "nombre_grupo", "demanda_media", "mae_modelo", "mae_semana", "cobertura_pct"]
if "distrito" in vista.columns:
    columnas.insert(1, "distrito")

st.dataframe(
    vista[columnas],
    hide_index=True,
    width="stretch",
    column_config={
        "zona": "Zona",
        "distrito": "Distrito",
        "nombre_grupo": "Tipo de zona",
        "demanda_media": st.column_config.NumberColumn("Viajes por hora (media)", format="%.1f"),
        "mae_modelo": st.column_config.NumberColumn("Error del modelo", format="%.1f"),
        "mae_semana": st.column_config.NumberColumn("Error «hace una semana»", format="%.1f"),
        "cobertura_pct": st.column_config.ProgressColumn(
            "Cobertura 10%–90%", format="%.0f%%", min_value=0, max_value=100),
    },
)

fig, ax = plt.subplots(figsize=(8, 3))
sns.histplot(data=por_zona, x="cobertura", bins=12, color=VERDE, ax=ax)
ax.axvline(0.8, color=NARANJA, linestyle="--", label="Objetivo (80%)")
ax.set(title="¿Cuántas zonas cubren el 80%?", xlabel="Cobertura de la zona", ylabel="Número de zonas")
ax.legend(frameon=False)
st.pyplot(fig)
plt.close(fig)

# Limitaciones
cmin, cmax, cmed = por_zona["cobertura"].min(), por_zona["cobertura"].max(), por_zona["cobertura"].mean()
bajo80 = int((por_zona["cobertura"] < 0.8).sum())
no_gana = int((~por_zona["gana_a_semana"]).sum())
dias_entrena = (pd.Timestamp(ficha["entrenamiento"]["hasta"]) - pd.Timestamp(ficha["entrenamiento"]["desde"])).days + 1
chico = grupos.loc[grupos["zonas"].idxmin()]

with st.container(border=True):
    st.subheader(":material/warning: Limitaciones")
    st.markdown(
        f"- **Pocos datos de entrenamiento:** solo Pocos días de enero de 2024. "
        f"No conoce data histórica de otros años en el mismo período.\n"
        f"- **Solo usa la historia de la propia demanda y el calendario** (retrasos de 1, 2 y 7 días, "
        f"hora, día de la semana y feriados): **no sabe de clima, conciertos ni cierres de calles**.\n"
        f"- **Cobertura variable:** el rango 10%–90% acierta {ficha['cobertura_80']:.1%} en general, "
        f"pero por zona va de {cmin:.0%} a {cmax:.0%} (promedio {cmed:.0%}); "
        f"{bajo80} de {len(por_zona)} zonas quedan por debajo del 80%.\n"
        f"- **Ventaja pequeña:** solo gana a «igual que hace una semana» en "
        f"{len(por_zona) - no_gana} de {len(por_zona)} zonas.\n"
        f"- **Grupos pequeños:** «{chico['nombre_grupo']}» tiene solo {int(chico['zonas'])} zonas, "
        f"así que sus cifras son poco concluyentes.\n"
        f"- **Alcance:** solo hay pronóstico para {ficha['n_zonas']} zonas."
    )

with st.container(border=True):
    st.subheader(":material/block: Usos no recomendados")
    st.markdown(
        "- **Decidir qué conductores trabajan** o evaluar su desempeño con este pronóstico: "
        "el modelo no conoce sus condiciones reales de trabajo.\n"
        f"- **Recortar el servicio en zonas de baja demanda** (como «{peor['nombre_grupo']}», "
        f"donde el error relativo es mayor): los datos son solo de taxis amarillos; "
        "poca demanda de taxi amarillo no significa poca necesidad de transporte.\n"
        "- **Usarlo más de un día hacia adelante:** está entrenado para pronosticar el día siguiente.\n"
        "- **Tratar el pronóstico como una cifra exacta:** es una estimación con un rango; "
        "úsalo para planificar, no para garantizar cuántos viajes habrá."
    )

# Calificación
st.divider()
st.markdown("**¿Te resultó clara esta ficha?**")
voto = st.feedback("stars", key="feedback_ficha")
if voto is not None and st.session_state.get("ultimo_voto") != voto:
    st.session_state["ultimo_voto"] = voto
    st.toast(f"¡Gracias! Calificaste con {voto + 1} de 5 estrellas.", icon=":material/thumb_up:")