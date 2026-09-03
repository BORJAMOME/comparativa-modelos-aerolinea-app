"""Carga de datos y artefactos del modelo, con cache de Streamlit."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"

CLASS_ORDER = ["Basico", "Frecuente", "Premium"]
CLASS_LABELS = {"Basico": "Básico", "Frecuente": "Frecuente", "Premium": "Premium"}
CLASS_COLORS = {"Basico": "#4A628E", "Frecuente": "#273A5F", "Premium": "#6E7F5B"}

MODEL_COLORS = {
    "Regresión Logística": "#B9C5D6",
    "Random Forest": "#4A628E",
    "Gradient Boosting": "#1D2638",
}

FEATURE_LABELS = {
    "edad": "Edad",
    "frecuencia_viaje_anual": "Vuelos al año",
    "distancia_media_km": "Distancia media (km)",
    "antelacion_media_dias": "Antelación de reserva (días)",
    "gasto_anual_eur": "Gasto anual (€)",
    "retrasos_ultimos_12m": "Retrasos (12m)",
    "cancelaciones_ultimos_12m": "Cancelaciones (12m)",
    "equipaje_medio_por_vuelo": "Equipaje medio por vuelo",
    "no_shows_ultimos_12m": "No-shows (12m)",
    "satisfaccion_media": "Satisfacción media (0-10)",
    "quejas_ultimos_12m": "Quejas (12m)",
    "antiguedad_programa_meses": "Antigüedad en el programa (meses)",
    "vuelos_largo_radio_anual": "Vuelos de largo radio al año",
    "vuelos_negocio_anual": "Vuelos de negocio al año",
    "uso_codigo_promocional": "Usa código promocional",
    "numero_maletas_max_12m": "Máx. maletas facturadas (12m)",
    "dia_preferido_viaje": "Día preferido de viaje",
    "temperatura_ciudad_residencia": "Temperatura media de residencia (°C)",
    "canal_registro": "Canal de registro",
    "aeropuerto_preferido": "Aeropuerto preferido",
}


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(ARTIFACTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name, **kwargs)


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    slugs = load_json("model_slugs.json")
    return {name: joblib.load(ARTIFACTS / f"model_{slug}.joblib") for name, slug in slugs.items()}


def artifacts_ready() -> bool:
    return (ARTIFACTS / "dataset_stats.json").exists() and (ARTIFACTS / "model_gb.joblib").exists()
