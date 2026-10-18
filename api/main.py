from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

BASE = Path(__file__).parent

VARIABLES = ["lag_24", "lag_48", "lag_167", "lag_168", "lag_169",
             "media_dia_ant", "hora", "dia_semana", "feriado", "zona"]
CUANTILES = np.round(np.arange(0.1, 0.91, 0.1), 1)
NOMBRES_Q = [f"p{int(round(q * 100))}" for q in CUANTILES]

modelo = xgb.XGBRegressor(enable_categorical=True)
modelo.load_model(BASE / "modelo_pronostico.ubj")

demanda = pd.read_parquet(BASE / "demanda.parquet")
demanda["dia"] = demanda["timestamp"].dt.date
demanda["nombre"] = demanda["nombre"].fillna("Zona " + demanda["PULocationID"].astype(str))
demanda["distrito"] = demanda["distrito"].fillna("Sin distrito")
# Mismas categorías (y mismo orden) que en el entrenamiento
DTIPO_ZONA = "int64"   

ids = sorted(int(z) for z in demanda["PULocationID"].unique())
demanda["zona"] = pd.Categorical(
    demanda["PULocationID"].astype(DTIPO_ZONA),
    categories=pd.Index(ids, dtype=DTIPO_ZONA),
)

FECHAS = sorted(str(d) for d in demanda["dia"].unique())

app = FastAPI(title="API de pronóstico de demanda - Taxis NYC")


@app.get("/")
def raiz():
    return {"estado": "en línea"}


@app.get("/info")
def info():
    zonas = (demanda[["PULocationID", "nombre", "distrito"]]
             .drop_duplicates().sort_values("nombre"))
    return {
        "zonas": [{"id": int(r.PULocationID), "nombre": r.nombre, "distrito": r.distrito}
                  for r in zonas.itertuples()],
        "fechas": FECHAS,
        "cuantiles": NOMBRES_Q,
    }


@app.get("/pronostico")
def pronostico(zona_id: int, fecha: date):
    filas = (demanda[(demanda["PULocationID"] == zona_id) & (demanda["dia"] == fecha)]
             .sort_values("timestamp"))
    if filas.empty:
        raise HTTPException(status_code=404, detail="No hay datos para esa zona o fecha")

    q = np.sort(modelo.predict(filas[VARIABLES]), axis=1).clip(min=0)

    horas = []
    for i, (_, f) in enumerate(filas.iterrows()):
        horas.append({
            "hora": int(f["hora"]),
            "real": None if pd.isna(f["viajes"]) else int(f["viajes"]),
            "hace_una_semana": None if pd.isna(f["lag_168"]) else int(f["lag_168"]),
            "cuantiles": {n: round(float(v), 1) for n, v in zip(NOMBRES_Q, q[i])},
        })

    return {
        "zona": {"id": int(zona_id), "nombre": filas["nombre"].iloc[0],
                 "distrito": filas["distrito"].iloc[0]},
        "fecha": str(fecha),
        "horas": horas,
    }
