import streamlit as st

st.set_page_config(page_title="Taxis NYC", layout="wide")


def url_por_defecto():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return "https://taxis-nyc.onrender.com"

if "api_url" not in st.session_state:
    st.session_state["api_url"] = url_por_defecto()

pagina = st.navigation({
    "Planificar": [
        st.Page("paginas/pronostico.py", title="Pronóstico",
                icon=":material/query_stats:", default=True),
        st.Page("paginas/decision.py", title="Decisión",
                icon=":material/local_taxi:"),
    ],
    "Entender": [
        st.Page("paginas/segmentacion.py", title="Segmentación de zonas",
                icon=":material/bubble_chart:"),
    ],
})

with st.sidebar:
    st.divider()
    st.markdown("### 🚕 Taxis NYC")
    st.caption("Demanda de taxis amarillos, enero de 2024 (NYC TLC).")
    with st.popover("Configurar API", icon=":material/settings:"):
        st.text_input("URL de la API", key="api_url",
                      help="Local: http://localhost:8000 · Publicada: la URL de Render")

pagina.run()