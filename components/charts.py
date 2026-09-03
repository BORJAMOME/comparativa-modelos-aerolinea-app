"""Figuras Plotly. Mismo sistema de color que el resto del portfolio:
azules marino para lo estructural, verde/rojo solo para lo semántico.
Los 3 modelos se colorean por "profundidad" (claro→oscuro = más simple→
más complejo), nunca con colores arbitrarios sin significado."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
NEGATIVE = "#C2412E"
SUPPORT = "#B8783C"
FONT = "Arial, Helvetica, sans-serif"

MODEL_COLOR = {"Regresión Logística": NAVY4, "Random Forest": NAVY3, "Gradient Boosting": INK}
CLASS_COLOR = {"Basico": NAVY4, "Frecuente": NAVY3, "Premium": POSITIVE}


def _base_layout(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        hovermode="closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def class_distribution(class_dist: dict, class_labels: dict) -> go.Figure:
    classes = list(class_dist.keys())
    labels = [class_labels.get(c, c) for c in classes]
    colors = [CLASS_COLOR.get(c, MUTED) for c in classes]
    fig = go.Figure(go.Bar(
        x=labels, y=list(class_dist.values()), marker_color=colors,
        text=[f"{v}%" for v in class_dist.values()], textposition="outside",
    ))
    fig.update_yaxes(title_text="% de clientes")
    return _base_layout(fig, height=360, legend=False)


def outlier_comparison(df: pd.DataFrame, feature_labels: dict, top_n: int = 8) -> go.Figure:
    """Grouped bar: top_n variables con más outliers detectados (por IQR), 3 métodos."""
    d = df.sort_values("IQR", ascending=False).head(top_n).iloc[::-1]
    labels = [feature_labels.get(v, v) for v in d["variable"]]
    fig = go.Figure()
    methods = [("IQR", NAVY4), ("Z-score", NAVY3), ("Percentiles P1/P99", INK)]
    for method, color in methods:
        fig.add_trace(go.Bar(y=labels, x=d[method], name=method, orientation="h", marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Nº de outliers detectados")
    return _base_layout(fig, height=440)


def correlation_heatmap(corr: pd.DataFrame, labels: dict) -> go.Figure:
    cols = list(corr.columns)
    nice = [labels.get(c, c) for c in cols]
    z = corr.values
    fig = go.Figure(go.Heatmap(
        z=z, x=nice, y=nice, zmin=-1, zmax=1,
        colorscale=[[0, NEGATIVE], [0.5, "#FBFBFB"], [1, NAVY2]],
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family=FONT, color=INK, size=10))
    fig.update_xaxes(tickangle=-40)
    return fig


def cv_comparison(cv_df: pd.DataFrame, baseline_f1: float) -> go.Figure:
    """Grouped bar: Accuracy vs F1-macro por modelo, con línea de baseline."""
    d = cv_df.sort_values("f1_macro_media", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=d["modelo"], x=d["accuracy_media"], name="Accuracy (CV)", orientation="h",
        marker_color=NAVY4,
        error_x=dict(type="data", array=d["accuracy_std"], color=MUTED, thickness=1),
    ))
    fig.add_trace(go.Bar(
        y=d["modelo"], x=d["f1_macro_media"], name="F1-macro (CV)", orientation="h",
        marker_color=INK,
        error_x=dict(type="data", array=d["f1_macro_std"], color=MUTED, thickness=1),
    ))
    fig.add_vline(x=baseline_f1, line_dash="dot", line_color=NEGATIVE,
                  annotation_text="baseline", annotation_font_color=NEGATIVE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Puntuación (validación cruzada 5-fold)", range=[0, 1])
    return _base_layout(fig, height=320)


def test_comparison(test_df: pd.DataFrame) -> go.Figure:
    d = test_df.sort_values("f1_macro", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["modelo"], x=d["accuracy"], name="Accuracy (test)",
                          orientation="h", marker_color=NAVY4))
    fig.add_trace(go.Bar(y=d["modelo"], x=d["f1_macro"], name="F1-macro (test)",
                          orientation="h", marker_color=INK))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Puntuación (hold-out test, 300 clientes)", range=[0, 1])
    return _base_layout(fig, height=320)


def confusion_matrix(matrix: list, labels: list, class_labels: dict) -> go.Figure:
    z = np.array(matrix)
    nice_labels = [class_labels.get(l, l) for l in labels]
    fig = go.Figure(go.Heatmap(
        z=z, x=nice_labels, y=nice_labels,
        colorscale=[[0, "#FBFBFB"], [1, NAVY2]],
        text=z, texttemplate="%{text}", textfont=dict(size=13),
        showscale=False,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=11),
        xaxis=dict(title="Predicho", side="bottom"),
        yaxis=dict(title="Real", autorange="reversed"),
    )
    return fig


def feature_importance(df: pd.DataFrame, feature_labels: dict, top_n: int = 10,
                        color: str = INK) -> go.Figure:
    d = df.head(top_n).iloc[::-1].copy()
    d["clean"] = d["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
    labels = [feature_labels.get(f, f) for f in d["clean"]]
    fig = go.Figure(go.Bar(x=d["importancia"], y=labels, orientation="h", marker_color=color))
    fig.update_xaxes(title_text="Importancia")
    return _base_layout(fig, height=420, legend=False)


def odds_ratios_class(odds_df: pd.DataFrame, class_name: str, feature_labels: dict,
                       top_n: int = 8) -> go.Figure:
    """Top variables que más alejan la probabilidad de `class_name` de 1 (odds ratio),
    en escala log2 para que subir/bajar sea visualmente simétrico."""
    d = odds_df[["feature", class_name]].copy()
    d = d[d["feature"].str.startswith("num__")]
    d["clean"] = d["feature"].str.replace("num__", "", regex=False)
    d["log2_or"] = np.log2(d[class_name])
    d = d.reindex(d["log2_or"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    labels = [feature_labels.get(f, f) for f in d["clean"]]
    colors = [POSITIVE if v > 0 else NEGATIVE for v in d["log2_or"]]
    fig = go.Figure(go.Bar(x=d["log2_or"], y=labels, orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_color=LINE)
    fig.update_xaxes(title_text=f"&#8592; menos probable Premium más probable &#8594;"
                      if class_name == "Premium" else "log2(odds ratio)")
    return _base_layout(fig, height=380, legend=False)


def playground_agreement(results: dict, class_labels: dict) -> go.Figure:
    """Grouped bar: para cada modelo, probabilidad asignada a cada clase —
    permite ver de un vistazo si los 3 modelos están de acuerdo o no."""
    models = list(results.keys())
    classes = ["Basico", "Frecuente", "Premium"]
    fig = go.Figure()
    for cls in classes:
        fig.add_trace(go.Bar(
            x=models, y=[results[m]["proba"].get(cls, 0) * 100 for m in models],
            name=class_labels.get(cls, cls), marker_color=CLASS_COLOR.get(cls, MUTED),
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Probabilidad (%)", range=[0, 100])
    return _base_layout(fig, height=380)
