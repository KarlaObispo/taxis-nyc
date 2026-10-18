import json
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de segmentación de zonas de taxi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Carga al iniciar ---
modelo = joblib.load("modelo_zonas.joblib")
with open("info_modelo.json", encoding="utf-8") as f:
    info = json.load(f)
with open("silueta.json") as f:
    silueta = json.load(f)

zonas = pd.read_csv("zonas_resumen.csv").set_index("zona_id")
perfil = pd.read_csv("perfil_horario.csv").set_index("zona_id")
perfil.columns = range(24)

VARIABLES = info["variables"]


def filtrar(grupos: Optional[List[int]]):
    return zonas if not grupos else zonas[zonas["grupo"].isin(grupos)]


@app.get("/")
def raiz():
    return {"variables": VARIABLES, "k": info["k"]}


@app.get("/grupos")
def grupos():
    """Una tarjeta por grupo: zonas, viajes/día, distancia mediana, % de noche."""
    salida = []
    for g, df_g in zonas.groupby("grupo"):
        salida.append({
            "grupo": int(g),
            "nombre": info["grupos"][str(g)]["nombre"],
            "descripcion": info["grupos"][str(g)]["descripcion"],
            "n_zonas": int(len(df_g)),
            "viajes_dia": int(df_g["viajes_dia"].sum()),
            "distancia_mediana": round(float(df_g["distancia_mediana"].median()), 1),
            "pct_noche": round(float(df_g["pct_22_5"].mean()), 3),
        })
    return salida


@app.get("/puntos-pca")
def puntos_pca(grupos: Optional[List[int]] = Query(default=None)):
    df_p = filtrar(grupos).reset_index()
    return df_p[["zona_id", "nombre", "pc1", "pc2", "grupo", "nombre_grupo", "viajes_dia"]].to_dict("records")


@app.get("/zonas")
def tabla_zonas(grupos: Optional[List[int]] = Query(default=None), limite: int = 50):
    df_z = filtrar(grupos).sort_values("viajes_dia", ascending=False).head(limite)
    filas = []
    for zona_id, r in df_z.iterrows():
        filas.append({
            "zona_id": int(zona_id),
            "zona": r["nombre"],
            "distrito": r["distrito"],
            "grupo": int(r["grupo"]),
            "nombre_grupo": r["nombre_grupo"],
            "viajes_dia": int(r["viajes_dia"]),
            "distancia_mediana": round(float(r["distancia_mediana"]), 1),
            "pct_noche": round(float(r["pct_22_5"]), 4),
            "propina_mediana": round(float(r["propina_mediana"]), 2),
            "perfil_24h": [round(float(v), 4) for v in perfil.loc[zona_id].values],
        })
    return filas


@app.get("/zona/{zona_id}")
def detalle_zona(zona_id: int):
    if zona_id not in zonas.index:
        raise HTTPException(status_code=404, detail="Zona no encontrada")
    r = zonas.loc[zona_id]
    return {
        "zona_id": zona_id,
        "nombre": r["nombre"],
        "distrito": r["distrito"],
        "grupo": int(r["grupo"]),
        "nombre_grupo": r["nombre_grupo"],
        "descripcion": r["descripcion"],
    }


@app.get("/perfil-horario")
def perfil_horario(zona_ids: List[int] = Query(...)):
    faltan = [z for z in zona_ids if z not in perfil.index]
    if faltan:
        raise HTTPException(status_code=404, detail=f"Zonas sin perfil: {faltan}")
    return [
        {"zona_id": z, "zona": zonas.loc[z, "nombre"], "valores": [float(v) for v in perfil.loc[z].values]}
        for z in zona_ids
    ]


@app.get("/silueta")
def get_silueta():
    return silueta


@app.post("/predecir")
def predecir(datos: dict):
    faltantes = [v for v in VARIABLES if v not in datos]
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan variables: {faltantes}")
    X = pd.DataFrame([datos])[VARIABLES]
    grupo = int(modelo["kmeans"].predict(modelo["scaler"].transform(X))[0])
    g = info["grupos"][str(grupo)]
    return {"grupo": grupo, "nombre_grupo": g["nombre"], "descripcion": g["descripcion"]}
