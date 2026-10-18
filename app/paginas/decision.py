import numpy as np
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
VERDE, NARANJA, ROJO = "#2f6f62", "#d9822b", "#b03a2e"
ZONA_POR_DEFECTO = 161  # Midtown Center


#  Acceso a la API
def obtener_api_url():
    url = st.session_state.get("api_url")
    if url:
        return url.rstrip("/")
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return "http://localhost:8000"


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


# Desicion
def costo_por_hora(enviados, real, costo_ocioso, costo_perdido):
    """Taxis de más cuestan 'ocioso'; pasajeros sin taxi cuestan 'perdido'."""
    return (np.maximum(enviados - real, 0) * costo_ocioso
            + np.maximum(real - enviados, 0) * costo_perdido)


def taxis_para_cuantil(df, columnas_q, niveles, cuantil):
    """Valor del cuantil pedido en cada hora (interpola entre los que entrega la API)."""
    valores = df[columnas_q].to_numpy()
    return np.array([np.interp(cuantil, niveles, fila) for fila in valores])


@st.fragment
def panel(df, columnas_q, niveles, titulo, zona_id, fecha):
    # ----- Controles -----
    c1, c2 = st.columns(2)
    costo_ocioso = c1.slider("Costo de un taxi sin pasajero durante una hora (USD)",
                             1, 100, 10, key="dec_costo_ocioso",
                             help="Lo que pierdes cuando un taxi espera una hora sin llevar a nadie.")
    costo_perdido = c2.slider("Costo de un pasajero que no encontró taxi (USD)",
                              1, 200, 40, key="dec_costo_perdido",
                              help="Lo que pierdes cuando alguien quería un taxi y no llego.")

    # Cuantil óptimo 
    cuantil = costo_perdido / (costo_perdido + costo_ocioso)
    razon = costo_perdido / costo_ocioso

    if razon > 1:
        st.markdown(f"Perder un pasajero cuesta **{razon:.1f} veces** lo que cuesta un taxi ocioso. "
                    f"Como quedarse corto sale más caro que pasarse, conviene enviar **más** que el "
                    f"pronóstico central: la regla óptima es cubrir el **cuantil {cuantil:.0%}** de la demanda.")
    elif razon < 1:
        st.markdown(f"Un taxi ocioso cuesta **{1 / razon:.1f} veces** lo que cuesta perder un pasajero. "
                    f"Como pasarse sale más caro que quedarse corto, conviene enviar **menos** que el "
                    f"pronóstico central: la regla óptima es cubrir el **cuantil {cuantil:.0%}** de la demanda.")
    else:
        st.markdown("Los dos errores cuestan lo mismo, así que la regla óptima es enviar "
                    "el pronóstico central (**cuantil 50%**).")

    st.latex(rf"\text{{cuantil óptimo}} = \frac{{C_p}}{{C_p + C_o}} = "
             rf"\frac{{{costo_perdido}}}{{{costo_perdido} + {costo_ocioso}}} = {cuantil:.2f}")

    if cuantil < niveles.min() or cuantil > niveles.max():
        st.info(f"El modelo solo entrega cuantiles de {niveles.min():.0%} a {niveles.max():.0%}. "
                f"Para este cociente se usa el más cercano.")

    # Taxis por hora 
    enviados = np.round(taxis_para_cuantil(df, columnas_q, niveles, cuantil))
    central = np.round(df["p50"].to_numpy())
    semana = np.round(df["hace_una_semana"].to_numpy())
    etiqueta_q = f"p{round(cuantil * 100)}"
    tiene_real = df["real"].notna().all()

    # Gráfico (seaborn) 
    graf = pd.DataFrame({"hora": df["hora"], "taxis": enviados, "central": central, "real": df["real"]})
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.barplot(data=graf, x="hora", y="taxis", color=VERDE, ax=ax,
                label=f"Taxis recomendados ({etiqueta_q})")
    sns.lineplot(data=graf, x="hora", y="central", ax=ax, color=NARANJA, linewidth=2.2,
                 label="Pronóstico central (p50)")
    if tiene_real:
        sns.scatterplot(data=graf, x="hora", y="real", ax=ax, color=ROJO, s=40,
                        label="Demanda real", zorder=3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set(xlabel="Hora", ylabel="Taxis / viajes", title=titulo)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    st.pyplot(fig)
    plt.close(fig)

    # ----- Costo de cada regla 
    if tiene_real:
        real = df["real"].to_numpy()
        costo_rec = costo_por_hora(enviados, real, costo_ocioso, costo_perdido).sum()
        costo_p50 = costo_por_hora(central, real, costo_ocioso, costo_perdido).sum()
        costo_sem = costo_por_hora(semana, real, costo_ocioso, costo_perdido).sum()

        def delta(costo):
            if costo_rec <= 0:
                return None
            return f"{(costo / costo_rec - 1) * 100:+.0f}% vs. la recomendación"

        st.markdown("**Si este día se hubiera decidido así, el costo habría sido:**")
        m1, m2, m3 = st.columns(3)
        with m1.container(border=True):
            st.metric(f"Recomendación ({etiqueta_q})", f"${costo_rec:,.0f}")
        with m2.container(border=True):
            st.metric("Enviar el pronóstico central (p50)", f"${costo_p50:,.0f}",
                      delta=delta(costo_p50), delta_color="inverse")
        with m3.container(border=True):
            st.metric("Enviar lo de hace una semana", f"${costo_sem:,.0f}",
                      delta=delta(costo_sem), delta_color="inverse")

        taxis_ociosos = np.maximum(enviados - real, 0).sum()
        sin_taxi = np.maximum(real - enviados, 0).sum()
        st.caption(f"Con la recomendación: {taxis_ociosos:,.0f} horas-taxi ociosas "
                   f"y {sin_taxi:,.0f} pasajeros sin taxi.")
    else:
        st.info("Este día todavía no tiene viajes reales (es un pronóstico puro), así que no se puede "
                "calcular el costo de cada regla. El plan de taxis sí se puede descargar.")

    # plan
    plan = pd.DataFrame({
        "hora": df["hora"],
        "taxis_recomendados": enviados.astype(int),
        "pronostico_central_p50": central.astype(int),
        "hace_una_semana": semana.astype(int),
        "cuantil_usado": round(cuantil, 3),
    })
    st.download_button("Descargar plan de taxis por hora",
                       plan.to_csv(index=False).encode("utf-8"),
                       file_name=f"plan_taxis_{zona_id}_{fecha}.csv", mime="text/csv",
                       key="dec_descarga")

    # Explicación 
    with st.expander("¿Por qué no enviar simplemente el pronóstico central?"):
        i = int(np.argmax(central))
        st.markdown(
            f"El pronóstico central (p50) es el punto medio: la mitad de las veces la demanda "
            f"real termina **por encima** de él. Si envías exactamente esa cantidad, "
            f"te quedarás corto cerca de la mitad de las horas.\n\n"
            f"Cuando perder un pasajero cuesta más que tener un taxi parado, ese error sale caro, "
            f"y conviene tener algunos taxis de más. Por ejemplo, a las {int(df['hora'].iloc[i]):02d}:00 "
            f"el pronóstico central es de {central[i]:,.0f} viajes, pero la recomendación envía "
            f"{enviados[i]:,.0f} taxis. Si los costos se invirtieran, la recomendación sería enviar "
            f"menos que el pronóstico central."
        )


# Página
st.title("¿Cuántos taxis enviar?")
st.caption("El pronóstico no es la decisión. La cantidad correcta depende de qué cuesta más: "
           "un taxi esperando sin pasajero o un pasajero que no encuentra taxi.")

url = obtener_api_url()
try:
    with st.spinner("Conectando con la API (la primera vez puede tardar hasta un minuto)..."):
        info = leer_info(url)
except requests.RequestException:
    st.error("No pude conectarme a la API. Revisa la URL en «Configurar API» de la barra lateral "
             "o espera unos segundos si el servicio estaba dormido.")
    st.stop()

zonas = sorted(info["zonas"], key=lambda z: z["nombre"])
etiquetas = {z["id"]: f'{z["nombre"]} ({z["distrito"]})' for z in zonas}
ids = list(etiquetas)
zona_id = st.selectbox("Zona", ids, format_func=etiquetas.get, key="dec_zona",
                       index=ids.index(ZONA_POR_DEFECTO) if ZONA_POR_DEFECTO in ids else 0)

fechas = info["fechas"]
ultima = fechas[-1]
fecha = st.radio("Fecha", fechas, format_func=lambda f: etiqueta_fecha(f, ultima),
                 horizontal=True, key="dec_fecha")

try:
    resp = leer_pronostico(url, zona_id, fecha)
except requests.HTTPError as e:
    if e.response is not None and e.response.status_code == 404:
        st.warning(e.response.json().get("detail", "No hay datos para esa zona o fecha."))
        st.stop()
    st.error("La API respondió con un error. Intenta de nuevo.")
    st.stop()

df = a_tabla(resp)
columnas_q = info["cuantiles"]                                   # ["p10", ..., "p90"]
niveles = np.array([int(c[1:]) / 100 for c in columnas_q])       # [0.1, ..., 0.9]
titulo = f'{resp["zona"]["nombre"]} · {etiqueta_fecha(fecha, ultima)}'

panel(df, columnas_q, niveles, titulo, zona_id, fecha)