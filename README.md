# taxis-nyc

# Proyecto de ML pronostico y segmentacion

# Taxis NYC: pronóstico de demanda por zona

Pronóstico de viajes por hora de taxis de Nueva York (enero de 2024), con rango de incertidumbre, una app para explorarlo y una API que sirve el modelo.

URL DE LA APP https://karlaobispo-taxis-nyc-appstreamlit-app-yiwbxk.streamlit.app/
URL DE RENDER https://taxis-nyc.onrender.com (documentación en `/docs`)

## Qué hace

- **Segmentación:** agrupa las zonas en 4 tipos con K-Means, K elegido por la silueta. Y gráfico a dos ComponenteS Principal
- **Pronóstico:** XGBoost con cuantiles (p10 a p90) por zona y hora, para el día siguiente.
- **Decisión:** convierte el pronóstico en cuántos taxis enviar según el costo de un taxi ocioso y el de un pasajero sin taxi.
- **Ficha del modelo:** explica qué tan bien funciona y dónde no conviene usarlo.

## Datos y resultados

- Viajes de taxis amarillos NYC TLC, enero de 2024, 58 zonas. Delimitados por zonas con mayor a 3000 viajes
- Entrenamiento: 8 al 24 de enero. Prueba: 25 al 31. Pronóstico: 1 de febrero.
- Error del modelo: 11.6 viajes/hora, contra 11.88 de "igual que hace una semana" y 21.29 de "igual que ayer".
- El rango 10%–90% cubre 83.8% de los casos (ideal: cerca de 80%).
- La segmentación utiliza datos para zonas con más de 500 viajes

## Estructura

```
api/    API con FastAPI (se publica en Render)
app/    App con Streamlit
  streamlit_app.py   archivo principal
  paginas/           pronóstico, decisión, segmentación, ficha del modelo
  data/              archivos con las cifras de la ficha
```

## Endpoints de la API

- `/` estado de la API
- `/info` zonas, fechas y cuantiles disponibles
- `/pronostico?zona_id=4&fecha=2024-02-01` pronóstico por hora de una zona

## Correr la app en local

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

La URL de la API se lee de `st.secrets["API_URL"]`. Crea `.streamlit/secrets.toml`:

```toml
API_URL