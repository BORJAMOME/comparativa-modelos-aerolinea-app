"""
Comparativa de Modelos — Segmentación de Clientes de Aerolínea
Case study interactivo en Streamlit: tres algoritmos de clasificación
comparados de forma justa, del baseline a la decisión de producción.

Autor: Borja Mora Méndez
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from components import charts, ui
from utils.classifier import build_input_row, predict_all
from utils.data_loader import (CLASS_COLORS, CLASS_LABELS, FEATURE_LABELS, artifacts_ready,
                                load_csv, load_json, load_models)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Comparativa de Modelos · Segmentación Aerolínea",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(ROOT / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. "
        "Ejecuta `py -3.10 model/train.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

stats = load_json("dataset_stats.json")
playground_defaults = load_json("playground_defaults.json")
confusion_matrices = load_json("confusion_matrices.json")
classification_reports = load_json("classification_reports.json")

nulos_df = load_csv("nulos.csv")
outlier_df = load_csv("outlier_comparativa.csv")
correlacion_df = load_csv("correlacion.csv", index_col=0)
vif_df = load_csv("vif.csv")
leakage_df = load_csv("leakage.csv")
cv_df = load_csv("cv_results.csv")
test_df = load_csv("test_results.csv")
fi_rf = load_csv("feature_importance_rf.csv")
fi_gb = load_csv("feature_importance_gb.csv")
odds_df = load_csv("odds_ratios_lr.csv")
odds_df = odds_df.rename(columns={odds_df.columns[0]: "feature"})

models = load_models()

n_fmt = f"{stats['n_customers']:,}".replace(",", ".")
GANADOR = stats["modelo_ganador"]


def pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Clasificación multiclase</p>
      <h1 class="hero-title">Tres modelos compiten por decidir en qué caja cae cada cliente. Solo uno merece el puesto <em>de verdad</em>.</h1>
      <p class="hero-sub">Comparamos Regresión Logística, Random Forest y Gradient Boosting para clasificar a
      {n_fmt} clientes de una aerolínea en tres segmentos comerciales — con el mismo split, el mismo
      preprocesamiento y la misma validación cruzada, para que ganara el mejor modelo y no el más
      publicitado.</p>
      <div class="hero-meta">
        <span class="hero-pill">Borja Mora Méndez</span>
        <span class="hero-pill">Python · scikit-learn</span>
        <span class="hero-pill">Streamlit</span>
        <span class="hero-pill">{n_fmt} clientes</span>
      </div>
      <div class="hero-scroll-row">
        <a href="#contexto" class="hero-scroll">explorar el caso &#8595;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================ CONTEXTO ==
ui.section_open("contexto")
ui.eyebrow("Contexto")
ui.h2("El problema")
ui.lead(
    "Una aerolínea con programa de fidelización diferencia tres niveles de cliente — Básico, "
    "Frecuente y Premium — pero no tiene un criterio automático para asignar ese nivel. Sin él, "
    "Marketing lanza la misma campaña genérica a todo el mundo: ni fideliza al frecuente que está a "
    "punto de subir de nivel, ni reactiva al básico que se está desconectando."
)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": f"{stats['n_columns_original']}", "label": "variables originales"},
    {"num": "3", "label": "modelos comparados"},
    {"num": "3", "label": "segmentos a predecir"},
])
st.write("")
ui.question_block(
    "La pregunta de negocio",
    'Con el comportamiento de vuelo, gasto e incidencias disponible, '
    '<span class="accent">¿se puede predecir el segmento con fiabilidad suficiente para automatizar campañas</span>, '
    'y qué modelo merece decidirlo?',
    "No basta con que un modelo funcione: hay que demostrar que supera con claridad a una regla trivial, "
    "y comparar varios candidatos con el mismo criterio antes de elegir cuál pasa a producción.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(
    f"{n_fmt} clientes reales, {stats['n_columns_original']} variables agrupadas en demografía, "
    "comportamiento de vuelo, valor económico, incidencias, equipaje/satisfacción y fidelización. "
    "La variable objetivo, <b>segmento_cliente</b>, ya viene asignada por el negocio — el modelo "
    "aprende a reproducir ese criterio, no a inventar uno nuevo."
)
ui.kpi_grid([
    {"num": pct(stats["class_distribution"]["Basico"] / 100), "label": "Básico"},
    {"num": pct(stats["class_distribution"]["Frecuente"] / 100), "label": "Frecuente"},
    {"num": pct(stats["class_distribution"]["Premium"] / 100), "label": "Premium"},
    {"num": f"{len(nulos_df)}", "label": "variables con nulos"},
])
ui.pipeline(["Datos crudos", "Calidad y outliers", "Capado P1/P99", "Imputación + escalado (en pipeline)",
             "Split 80/20 estratificado", "3 modelos, mismo criterio"])

ui.eyebrow("Calidad del dato", muted=True)
ui.h3("¿Cuántos nulos hay, y dónde?")
st.dataframe(
    nulos_df.rename(columns={"variable": "Variable", "nulos": "Nulos", "pct_nulos": "% nulos"}),
    use_container_width=True, hide_index=True,
)
ui.finding(
    "Entre el 1,5% y el 3,5% de nulos en 5 variables — poco, pero suficiente para que un "
    "<code>dropna()</code> a lo bruto tirara el 10,8% de las filas. Se imputan dentro del pipeline "
    "(mediana en numéricas, moda en categóricas), ajustado solo sobre el conjunto de entrenamiento."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué nos dicen los datos?")
ui.lead(
    "Tres preguntas antes de entrenar nada: ¿hay valores extremos que puedan distorsionar el modelo?, "
    "¿hay variables que miden lo mismo dos veces?, ¿alguna variable delata la respuesta demasiado bien?"
)

ui.h3("¿Cuántos outliers hay, según el método que uses?")
st.plotly_chart(charts.outlier_comparison(outlier_df, FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    f"Los cuatro métodos no coinciden — normal, cada uno mide algo distinto. Se optó por "
    f"<b>capar (winsorizing) en los percentiles P1/P99</b>, la opción más conservadora: conserva el "
    f"100% de los {n_fmt} clientes pero evita que un valor extremo distorsione medias y modelos. "
    f"Isolation Forest, que sí mira todas las variables a la vez, señala un {pct(stats['n_outliers_iforest']/stats['n_customers'])} "
    "de clientes como atípicos globales — se calcula solo con fines de EDA, nunca se usa como variable del modelo."
)

ui.h3("¿Hay variables que miden lo mismo dos veces?")
st.plotly_chart(charts.correlation_heatmap(correlacion_df, FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
max_vif = vif_df.loc[vif_df["VIF"].idxmax()]
ui.finding(
    f"La correlación entre variables es prácticamente nula en casi todos los pares, y el <b>VIF</b> "
    f"(factor de inflación de varianza) confirma que no hay multicolinealidad real: la variable más "
    f"alta es <b>{FEATURE_LABELS.get(max_vif['variable'], max_vif['variable'])}</b> con VIF={max_vif['VIF']:.2f} "
    "— muy por debajo del umbral de alerta (5). Cada variable aporta información distinta, así que no "
    "hace falta descartar ninguna antes de modelar."
)

ui.h3("¿Alguna variable delata el segmento antes de tiempo?")
with st.expander("Ver el test de fuga de información (ANOVA F-test / eta²)"):
    st.dataframe(
        leakage_df.rename(columns={"variable": "Variable", "F_stat": "F", "p_valor": "p-valor", "eta2": "eta²"}),
        use_container_width=True, hide_index=True,
    )
top_leak = leakage_df.iloc[0]
ui.finding(
    f"Ninguna variable supera eta²=0.95 (separación casi perfecta, señal de fuga de información). La más "
    f"relacionada con el segmento es <b>{FEATURE_LABELS.get(top_leak['variable'], top_leak['variable'])}</b> "
    f"(eta²={top_leak['eta2']:.2f}) — una relación real y esperable, no una fuga: es justo el tipo de "
    "variable que debería influir en el segmento de un cliente frecuente."
)
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo se llegó al modelo")
ui.h2("El camino hasta la comparativa")
ui.lead(
    "Comparar modelos tiene una trampa: si cada uno se entrena y evalúa a su manera, la comparación no "
    "es justa. Este es el camino para evitarlo."
)
ui.story_steps([
    ("Partimos del dato capado, no del original",
     "El winsorizing (P1/P99) de la sección anterior alimenta el modelo — así el tratamiento de "
     "outliers deja de ser un ejercicio aislado de EDA y pasa a formar parte del pipeline real."),
    ("Excluimos explícitamente la señal de Isolation Forest",
     "Es un artefacto de EDA calculado sobre todo el dataset antes de separar train y test — no es una "
     "variable de negocio disponible para un cliente nuevo. Usarla sería hacer trampas."),
    ("Imputamos y escalamos dentro de un Pipeline",
     "Nada se imputa a mano ni fuera del flujo: la mediana/moda del imputador se calcula solo sobre "
     "entrenamiento, evitando que información de test se filtre al modelo."),
    ("Definimos un suelo mínimo: el baseline",
     f"Un clasificador que siempre predice la clase mayoritaria acierta el {pct(stats['baseline']['accuracy'])} "
     "de las veces. Ningún modelo real vale la pena si no lo supera con claridad."),
    ("Entrenamos los 3 candidatos con el mismo split y preprocesamiento",
     "Regresión Logística, Random Forest y Gradient Boosting, envueltos en el mismo Pipeline — la única "
     "diferencia entre ellos es el algoritmo, no la preparación de los datos."),
    ("Comparamos con validación cruzada antes de mirar el test",
     "Un único split 80/20 (300 clientes) tiene demasiada varianza para declarar un ganador. Los 3 "
     "modelos se comparan primero con 5-fold estratificado sobre el conjunto de entrenamiento."),
    ("Confirmamos en un test que ningún modelo vio nunca",
     "Solo al final, cada modelo se reentrena sobre todo el train y se evalúa una única vez sobre el "
     "20% que quedó completamente al margen — la prueba de que el ranking no es casualidad."),
])
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Cómo intenta resolverlo?")
ui.h2("Los 3 modelos, cara a cara")
ui.lead(
    "Tres algoritmos con lógicas muy distintas — un modelo lineal, un ensamble de árboles en paralelo, "
    "y árboles entrenados en secuencia — sobre el mismo split y el mismo preprocesamiento."
)

ui.h3("Validación cruzada (5-fold estratificado, solo sobre train)")
st.plotly_chart(charts.cv_comparison(cv_df, stats["baseline"]["f1_macro"]), use_container_width=True,
                 config={"displayModeBar": False})
best_cv = cv_df.iloc[0]
worst_cv = cv_df.iloc[-1]
ui.finding(
    f"<b>{best_cv['modelo']}</b> lidera en validación cruzada con F1-macro={best_cv['f1_macro_media']:.3f} "
    f"(±{best_cv['f1_macro_std']:.3f}) — un {(best_cv['f1_macro_media']/worst_cv['f1_macro_media']-1)*100:.0f}% "
    f"mejor que {worst_cv['modelo']} ({worst_cv['f1_macro_media']:.3f}). Los tres superan con claridad el "
    f"baseline ({stats['baseline']['f1_macro']:.3f}): hay señal real en los datos de comportamiento."
)

ui.h3("Confirmación en test hold-out (300 clientes que ningún modelo vio)")
st.plotly_chart(charts.test_comparison(test_df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    "El ranking en test coincide exactamente con el de validación cruzada — la señal es consistente, no "
    "un golpe de suerte de un único split. Eso da confianza para elegir el modelo ganador."
)

ui.h3("Matrices de confusión (test)")
cm_cols = st.columns(3)
for c, nombre in zip(cm_cols, ["Regresión Logística", "Random Forest", "Gradient Boosting"]):
    with c:
        st.markdown(f'<p class="co-body" style="font-weight:700; text-align:center;">{nombre}</p>',
                    unsafe_allow_html=True)
        cm = confusion_matrices[nombre]
        st.plotly_chart(charts.confusion_matrix(cm["matrix"], cm["labels"], CLASS_LABELS),
                         use_container_width=True, config={"displayModeBar": False})
gb_cm = confusion_matrices[GANADOR]["matrix"]
ui.finding(
    f"En {GANADOR}, la confusión se concentra casi siempre en la clase vecina: Básico se confunde con "
    f"Frecuente ({gb_cm[0][1]} casos) mucho más que con Premium ({gb_cm[0][2]} caso) — y Premium nunca se "
    "confunde con Básico (0 casos). El modelo se equivoca donde tiene sentido equivocarse: entre "
    "segmentos adyacentes, nunca entre los dos extremos."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("¿Por qué decide así?")
ui.h2("Explicabilidad")
ui.lead(
    "Cada modelo se puede interrogar de una forma distinta: la Regresión Logística da coeficientes "
    "directos por clase, Random Forest y Gradient Boosting dan qué variables mueven más sus árboles."
)

fi_col1, fi_col2 = st.columns(2)
with fi_col1:
    ui.h3("Random Forest — variables más usadas")
    st.plotly_chart(charts.feature_importance(fi_rf, FEATURE_LABELS, color=charts.NAVY3),
                     use_container_width=True, config={"displayModeBar": False})
with fi_col2:
    ui.h3(f"{GANADOR} — variables más usadas")
    st.plotly_chart(charts.feature_importance(fi_gb, FEATURE_LABELS, color=charts.INK),
                     use_container_width=True, config={"displayModeBar": False})
ui.finding(
    "Ambos ensambles coinciden en el podio: frecuencia de viaje, distancia media y gasto anual "
    "concentran más de la mitad de la importancia total. Pero hay un matiz honesto que hay que señalar: "
    "<b>distancia_media_km</b> pesa mucho más aquí que en el test de fuga de información (eta² bajo, "
    "sección anterior) — es un sesgo conocido de este tipo de importancia nativa, que tiende a "
    "sobrevalorar variables numéricas con muchos valores distintos frente a las categóricas o discretas."
)

ui.h3("Regresión Logística — ¿qué variables empujan hacia Premium?")
st.plotly_chart(charts.odds_ratios_class(odds_df, "Premium", FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    "A diferencia de los ensambles, aquí el signo importa: <b>gasto anual</b> y <b>frecuencia de viaje</b> "
    "empujan con fuerza hacia Premium, mientras que un gasto o frecuencia bajos empujan hacia Básico. "
    "Es la ventaja de un modelo lineal — el resultado se traduce a lenguaje de negocio sin intermediarios: "
    "\"cada euro adicional de gasto anual aumenta la probabilidad de Premium\", algo que un "
    "<code>feature_importance_</code> no puede afirmar por sí solo."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Pruébalo tú mismo")
ui.h2("Playground — ¿en qué segmento caería este cliente, según cada modelo?")
ui.lead(
    "Ajusta el comportamiento de un cliente hipotético. Los 3 modelos predicen en vivo, con la "
    "probabilidad que le asigna cada uno a cada segmento — para ver si están de acuerdo, o no."
)

num_defaults = playground_defaults["numeric"]
cat_options = playground_defaults["categorical_options"]
cat_defaults = playground_defaults["categorical_defaults"]
numeric_features = playground_defaults["numeric_features"]
categorical_features = playground_defaults["categorical_features"]

pg_left, pg_right = st.columns([1, 1.3], gap="large")
with pg_left:
    st.markdown("**Comportamiento de vuelo**")
    frecuencia = st.slider("Vuelos al año", 2, 12, int(num_defaults["frecuencia_viaje_anual"]["median"]))
    distancia = st.slider("Distancia media (km)", 300, 6700, int(num_defaults["distancia_media_km"]["median"]), step=50)
    largo_radio = st.slider("Vuelos de largo radio al año", 0, 5, int(num_defaults["vuelos_largo_radio_anual"]["median"]))
    negocio = st.slider("Vuelos de negocio al año", 0, 5, int(num_defaults["vuelos_negocio_anual"]["median"]))
    st.markdown("**Valor y fidelización**")
    gasto = st.slider("Gasto anual (€)", 185, 2800, int(num_defaults["gasto_anual_eur"]["median"]), step=25)
    antiguedad = st.slider("Antigüedad en el programa (meses)", 2, 180, int(num_defaults["antiguedad_programa_meses"]["median"]))
    st.markdown("**Canal**")
    canal = st.selectbox("Canal de registro", cat_options["canal_registro"],
                          index=cat_options["canal_registro"].index(cat_defaults["canal_registro"]))
    aeropuerto = st.selectbox("Aeropuerto preferido", cat_options["aeropuerto_preferido"],
                               index=cat_options["aeropuerto_preferido"].index(cat_defaults["aeropuerto_preferido"]))

user_values = {f: num_defaults[f]["median"] for f in numeric_features}
user_values.update({
    "frecuencia_viaje_anual": frecuencia, "distancia_media_km": distancia,
    "vuelos_largo_radio_anual": largo_radio, "vuelos_negocio_anual": negocio,
    "gasto_anual_eur": gasto, "antiguedad_programa_meses": antiguedad,
})
user_values["canal_registro"] = canal
user_values["aeropuerto_preferido"] = aeropuerto

X_row = build_input_row(user_values, numeric_features, categorical_features)
results = predict_all(models, X_row)
preds = {m: r["pred"] for m, r in results.items()}
unanime = len(set(preds.values())) == 1

with pg_right:
    badge_cols = st.columns(3)
    for c, nombre in zip(badge_cols, ["Regresión Logística", "Random Forest", "Gradient Boosting"]):
        pred = preds[nombre]
        proba = results[nombre]["proba"][pred]
        with c:
            st.markdown(
                f'<div class="co-card" style="text-align:center; padding:1.2rem .8rem; height:100%;">'
                f'<div class="kpi-label" style="font-size:.72rem;">{nombre}</div>'
                f'<div class="kpi-num" style="font-size:1.35rem; color:{CLASS_COLORS[pred]}; margin:.4rem 0 .2rem;">'
                f'{CLASS_LABELS[pred]}</div>'
                f'<div class="kpi-label">{pct(proba)} de confianza</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.plotly_chart(charts.playground_agreement(results, CLASS_LABELS), use_container_width=True,
                     config={"displayModeBar": False})

ui.h3("¿Están de acuerdo los tres modelos?")
if unanime:
    ui.finding(
        f"Los 3 modelos coinciden: <b>{CLASS_LABELS[list(preds.values())[0]]}</b>. Cuando el perfil del "
        "cliente es claro, hasta los modelos más simples llegan a la misma conclusión que el más complejo "
        "— la ventaja de un modelo sofisticado se nota sobre todo en los casos ambiguos, no en estos."
    )
else:
    ui.finding(
        "Los modelos <b>no están de acuerdo</b> — señal de que este cliente cae en una zona fronteriza "
        "entre dos segmentos. Es exactamente el tipo de caso donde la elección del modelo importa de "
        f"verdad: en producción, es {GANADOR} quien decide, por ser el que mejor generaliza en test."
    )
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("¿Funciona de verdad?")
ui.h2("Resultados")
best_test = test_df.iloc[0]
worst_test = test_df.iloc[-1]
ui.lead(
    f"<b>{best_test['modelo']}</b> gana en las dos evaluaciones — validación cruzada y test — con un "
    f"F1-macro de {best_cv['f1_macro_media']:.3f} en CV y {best_test['f1_macro']:.3f} en test, frente al "
    f"{pct(stats['baseline']['f1_macro'])} del baseline. Que el ranking coincida en ambas evaluaciones es "
    "la señal de que no es un golpe de suerte de un único split."
)

report_gb = classification_reports[GANADOR]
ui.h3(f"¿Dónde acierta más y menos {GANADOR}?")
rep_cols = st.columns(3)
for c, cls in zip(rep_cols, ["Basico", "Frecuente", "Premium"]):
    with c:
        row = report_gb[cls]
        st.markdown(
            f'<div class="co-card" style="height:100%;">'
            f'<div class="kpi-label" style="font-weight:700; color:{CLASS_COLORS[cls]};">{CLASS_LABELS[cls]}</div>'
            f'<div class="kpi-num" style="font-size:1.6rem; margin:.3rem 0;">{row["f1-score"]:.2f}</div>'
            f'<div class="kpi-label">F1-score · precisión {row["precision"]:.2f} · recall {row["recall"]:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
st.write("")
ui.finding(
    f"El segmento más difícil de acertar es <b>Frecuente</b> (F1={report_gb['Frecuente']['f1-score']:.2f}), "
    "justo el que está en medio — comparte comportamiento con ambos vecinos. Básico y Premium, los dos "
    f"extremos, son más fáciles de distinguir (F1={report_gb['Basico']['f1-score']:.2f} y "
    f"{report_gb['Premium']['f1-score']:.2f}). Tiene sentido de negocio: el punto de corte entre "
    "\"empieza a viajar mucho\" y \"ya es Premium\" es, por naturaleza, más difuso que los extremos."
)
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'<span class="accent-pos">{GANADOR}</span> acierta el segmento '
    f'<span class="accent-pos">{pct(best_test["accuracy"], 0)}</span> de las veces — '
    f'frente al <span class="accent-neg">{pct(stats["baseline"]["accuracy"], 0)}</span> de asignar siempre la clase mayoritaria.',
    quote='"Los tres modelos superan con claridad el baseline: la señal es real. La elección final ya no es solo técnica — depende de lo que necesite Marketing."',
)
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("¿Qué haríamos con esto?")
ui.h2("Decisiones que habilita")
ui.decision_flow(
    "Marketing necesita justificar cada campaña variable a variable, no solo confiar en una caja negra",
    "Desplegar Regresión Logística — sus odds ratios se traducen a lenguaje de negocio sin intermediarios",
    "Explicabilidad total en decisiones que un cliente puede cuestionar",
    "% de campañas justificables ante el cliente",
)
st.write("")
ui.decision_flow(
    f"{GANADOR} maximiza el acierto ({pct(best_test['accuracy'])} en test) y la explicabilidad pasa a segundo plano",
    f"Desplegar {GANADOR} como clasificador de producción",
    "Maximizar la precisión del segmento asignado, aceptando menos transparencia",
    "F1-macro en producción vs. el reportado en test",
)
st.write("")
ui.decision_flow(
    "Random Forest queda como punto medio: mejor que la logística, más fácil de inspeccionar que el boosting",
    "Usarlo como modelo de contraste o candidato de respaldo",
    "Tener una segunda opinión rápida de revisar cuando el modelo principal dude",
    "Tasa de desacuerdo entre modelos en producción",
)
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
lc1, lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown('<p class="limit-col-title">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<ul class="limit-list">
        <li>Clasificar automáticamente y con {pct(best_test['accuracy'])} de acierto, muy por encima del baseline.</li>
        <li>Distinguir con claridad los dos extremos (Básico y Premium) — casi nunca los confunde entre sí.</li>
        <li>Ofrecer una versión explicable (Regresión Logística) cuando Marketing necesita justificar la decisión.</li>
        <li>Señalar qué variables pesan más, abriendo la puerta a acciones comerciales dirigidas.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<ul class="limit-list">
        <li>Distinguir con la misma fiabilidad al segmento intermedio (Frecuente) — es el que más se confunde.</li>
        <li>Ser mejor que las variables con las que se entrena: si incidencias o satisfacción tienen sesgo o ruido, ningún modelo lo compensa.</li>
        <li>Mantenerse fiable si el comportamiento de vuelo cambia de forma estructural sin reentrenar.</li>
        <li>Sustituir el criterio comercial — da una probabilidad, no una sentencia irrevocable.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "El rendimiento depende por completo de la calidad de las variables de incidencias y satisfacción. "
    "Si su recogida en producción tiene sesgo o ruido — encuestas que solo responden los clientes más "
    "extremos, incidencias mal registradas — ningún modelo, por sofisticado que sea, lo compensa."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("Conclusión")
ui.lead(
    f"Los tres modelos superan con claridad el baseline ({pct(stats['baseline']['accuracy'])} de acierto de "
    f"la clase mayoritaria), confirmando que el comportamiento de vuelo tiene señal real. "
    f"<b>{GANADOR}</b> gana la comparativa tanto en validación cruzada como en test — pero la elección final "
    "no es solo técnica: depende de si Marketing necesita explicar cada decisión o solo acertarla."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/comparativa-modelos-aerolinea-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
