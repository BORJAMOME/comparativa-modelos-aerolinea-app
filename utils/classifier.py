"""Predice el segmento de un cliente hipotético en vivo con los 3 modelos
entrenados (mismos pipelines usados para reportar accuracy/F1 en test),
para poder comparar si están de acuerdo o no sobre el mismo cliente."""
import pandas as pd


def build_input_row(values: dict, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    row = {f: values[f] for f in numeric_features + categorical_features}
    return pd.DataFrame([row])


def predict_all(models: dict, X_row: pd.DataFrame) -> dict:
    """Devuelve {nombre_modelo: {"pred": clase, "proba": {clase: prob}}}."""
    results = {}
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_row)[0]
        classes = pipe.named_steps["modelo"].classes_
        proba_dict = dict(zip(classes, proba.tolist()))
        pred = max(proba_dict, key=proba_dict.get)
        results[name] = {"pred": pred, "proba": proba_dict}
    return results
