"""
Comparativa de Modelos — Segmentación de Clientes de Aerolínea
Reportaje digital de datos: tres algoritmos de clasificación comparados de forma justa,
del problema de negocio a la decisión.

Narrativa (beats):
  LEDE → 01 El problema → 02 Los datos → 03 Antes de modelar → 04 El método →
  05 La evidencia → 06 Lo que hemos aprendido → 07 Ponlo a prueba →
  08 Implicaciones → 09 Limitaciones → 10 Conclusión

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial
reutilizable); aquí solo hay contenido y datos.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import streamlit as st

from components import charts
from components import editorial as ed
from utils.classifier import build_input_row, predict_all
from utils.data_loader import (CLASS_COLORS, CLASS_LABELS, FEATURE_LABELS, artifacts_ready,
                                load_csv, load_json, load_models)

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales
# ya importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja
# un app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada
# ejecución lo evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ed)

ROOT = Path(__file__).resolve().parent
LECTURA = "13 min"
ACTUALIZADO = "Septiembre 2026"
FUENTE = "Elaboración propia"

st.set_page_config(
    page_title="Comparativa de Modelos · Segmentación Aerolínea",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Identidad del proyecto primero (paleta, tipografías); sistema editorial después
ed.load_css(ROOT, "style.css", "editorial.css")

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
dist = stats["class_distribution"]

best_cv, worst_cv = cv_df.iloc[0], cv_df.iloc[-1]
best_test = test_df.iloc[0]
report_gb = classification_reports[GANADOR]
gb_cm = confusion_matrices[GANADOR]["matrix"]
max_vif = vif_df.loc[vif_df["VIF"].idxmax()]
top_leak = leakage_df.iloc[0]
top_leak_name = FEATURE_LABELS.get(top_leak["variable"], top_leak["variable"])
odds_idx = odds_df.set_index("feature")
or_frec = odds_idx.loc["num__frecuencia_viaje_anual", "Premium"]
or_gasto = odds_idx.loc["num__gasto_anual_eur", "Premium"]
dist_eta2 = leakage_df.loc[leakage_df["variable"] == "distancia_media_km", "eta2"].iloc[0]
top3_share = min(fi_rf["importancia"].head(3).sum(), fi_gb["importancia"].head(3).sum())
PLOT = {"displayModeBar": False}


def es(value: float, decimals: int = 3) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value * 100, decimals)}%"


# ============================================================ LEDE ==
ed.skip_link("contexto")
ed.topbar(
    "Comparativa de Modelos · Aerolínea",
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("evidencia", "Evidencia"),
     ("aprendizajes", "Aprendizajes"), ("playground", "Playground"), ("implicaciones", "Implicaciones"),
     ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Clasificación de clientes",
    headline="¿Y si una aerolínea pudiera <em>reconocer</em> automáticamente qué tipo de cliente tiene delante?",
    deck=(
        "Un cliente que vuela una vez al año no debería recibir necesariamente la misma propuesta que uno "
        "que vuela todas las semanas. A partir de su comportamiento de vuelo, gasto y relación con la "
        "aerolínea, entrené y comparé tres modelos de Machine Learning para clasificar a cada cliente como "
        "<b>Básico, Frecuente o Premium</b>. El objetivo: comprobar si los datos permiten automatizar esa "
        "clasificación y <b>qué modelo lo hace mejor</b>."
    ),
    meta=[
        ("Autor", "Borja Mora Méndez"),
        ("Stack", "Python · scikit-learn · Streamlit"),
        ("Datos", f"{n_fmt} clientes · {stats['n_columns_original']} variables"),
        ("Actualizado", ACTUALIZADO),
        ("Lectura", LECTURA),
    ],
)

# ============================================================ 01 · EL PROBLEMA ==
ed.beat(
    "contexto", "01", "Contexto", "El problema",
    deck=(
        "La aerolínea tiene tres tipos de cliente: <b>Básico, Frecuente y Premium</b>. El reto es sencillo "
        "de plantear: si conocemos cómo vuela, cuánto gasta y cómo se relaciona con la compañía, ¿podemos "
        "utilizar esos datos para <b>identificar automáticamente</b> a qué segmento pertenece? Esto "
        "permitiría pasar de campañas genéricas a acciones más adaptadas a cada tipo de cliente."
    ),
)
ed.band(
    "La pregunta de negocio",
    '¿Podemos <span class="accent">predecir el segmento de un cliente a partir de su comportamiento</span> '
    "y hacerlo con suficiente fiabilidad como para utilizarlo en campañas comerciales?",
    "Para responderla, comparé tres modelos diferentes utilizando exactamente los mismos datos y "
    "criterios de evaluación.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Punto de partida", "Los datos",
    deck=(
        f"Partimos de <b>{n_fmt} clientes</b> y {stats['n_columns_original']} variables que describen "
        "diferentes aspectos de su relación con la aerolínea: frecuencia de vuelo, gasto, distancia "
        "recorrida, antigüedad, incidencias, satisfacción y fidelización. Lo que queremos predecir es "
        "<b>el segmento del cliente</b>: el modelo aprende a reconocer patrones en el comportamiento de "
        "clientes que ya conocemos para poder aplicarlos después a nuevos clientes."
    ),
)
ed.provenance([
    ("Dataset", "dataset_linea_aerea_multiclase_v2.xlsx"),
    ("Observaciones", f"{n_fmt} clientes"),
    ("Variables", f"{stats['n_columns_original']} ({stats['n_features_used']} predictoras + el segmento)"),
    ("Segmentos", f"Básico {pct(dist['Basico'] / 100, 0)} · Frecuente {pct(dist['Frecuente'] / 100, 0)} · "
                  f"Premium {pct(dist['Premium'] / 100, 0)}"),
    ("Incompletas", f"{len(nulos_df)} variables con valores vacíos"),
    ("Elaboración", FUENTE),
])
ed.passage(
    "El segmento, <code>segmento_cliente</code>, ya viene asignado por el negocio: el modelo aprende a "
    "reproducir ese criterio, no a inventar uno nuevo.",
)

ed.subhead("Antes de construir el modelo")
ed.passage(
    "Un modelo solo puede ser tan bueno como los datos con los que aprende. Antes de entrenar los "
    "modelos revisé tres aspectos: <b>datos incompletos</b>, <b>valores extremos</b> y variables que "
    "aporten <b>información duplicada</b> o demasiado cercana al resultado que queremos predecir.",
    tight=True,
)

ed.subhead("¿Cuántos datos faltan?", level="wide")
with ed.split("nulos", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Encontramos valores incompletos en {len(nulos_df)} variables. En lugar de eliminar esos "
            "clientes, completé los valores utilizando información disponible en el propio conjunto de "
            f"entrenamiento. Así <b>mantenemos los {n_fmt} clientes</b> y evitamos perder información "
            "innecesariamente. El tratamiento se realiza dentro del pipeline para evitar que información "
            "del conjunto de prueba influya en el entrenamiento."
        )
        ed.note(
            "<b>Detalle técnico.</b> Entre el 1,5% y el 3,5% de nulos por variable: poco, pero suficiente "
            "para que un <code>dropna()</code> descartara el 10,8% de las filas. Imputación con la mediana "
            "(numéricas) y la moda (categóricas), ajustada solo sobre entrenamiento."
        )
    with viz:
        st.dataframe(
            nulos_df.rename(columns={"variable": "Variable", "nulos": "Nulos", "pct_nulos": "% nulos"}),
            use_container_width=True, hide_index=True,
        )
        ed.caption("TAB. 01", "Variables con valores incompletos: número y porcentaje de nulos.", FUENTE)

# ============================================================ 03 · ANTES DE MODELAR ==
ed.beat(
    "comprobaciones", "03", "Antes de modelar", "¿Qué me dicen los datos?",
    deck=(
        "Resueltos los datos incompletos, quedan tres preguntas antes de entrenar nada: ¿hay clientes con "
        "<b>comportamientos extremos</b>? ¿Hay variables que cuentan prácticamente <b>lo mismo</b>? ¿Alguna "
        "variable le está <b>«chivando»</b> la respuesta al modelo?"
    ),
)

ed.subhead("¿Hay clientes con comportamientos extremos?", level="wide")
with ed.split("extremos", "5-7") as (txt, viz):
    with txt:
        ed.passage(
            "No todos los clientes tienen un comportamiento «normal»: algunos vuelan muchísimo más, gastan "
            "mucho más o recorren distancias muy superiores al resto. Estos valores pueden <b>influir "
            "demasiado</b> en un modelo, así que analizamos los posibles casos extremos antes de entrenarlo."
        )
        ed.insight(
            "En lugar de eliminar clientes, opté por <b>limitar los valores más extremos</b> entre los "
            f"percentiles P1 y P99. De esta forma conservamos los {n_fmt} clientes, pero evitamos que unos "
            "pocos valores extremos tengan un peso desproporcionado."
        )
        ed.note(
            "<b>Detalle técnico.</b> Los cuatro métodos no coinciden: es normal, cada uno mide algo distinto. "
            "Opté por capar (winsorizing) en los percentiles P1/P99, la opción más conservadora. Isolation "
            f"Forest, que sí mira todas las variables a la vez, señala un "
            f"{pct(stats['n_outliers_iforest'] / stats['n_customers'])} de clientes como atípicos globales; "
            "lo calculé solo con fines de exploración, nunca lo usé como variable del modelo."
        )
    with viz:
        st.plotly_chart(charts.outlier_comparison(outlier_df, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Las ocho variables con más valores atípicos, según tres métodos de detección "
                              "(IQR, Z-score y percentiles P1/P99).", FUENTE)

ed.subhead("¿Hay variables que cuentan prácticamente lo mismo?", level="wide")
with ed.split("duplicadas", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.correlation_heatmap(correlacion_df, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 02", "Correlación entre pares de variables numéricas.", FUENTE)
    with txt:
        ed.passage(
            "Si dos variables contienen información muy parecida, pueden aportar <b>poco valor adicional</b> "
            "al modelo. Por eso comprobé la relación entre las variables antes de entrenarlo."
        )
        ed.insight(
            "El resultado fue positivo: <b>no encontramos variables duplicadas</b> o excesivamente "
            "relacionadas entre sí. Esto significa que podemos mantener la información disponible sin "
            "necesidad de eliminar variables por este motivo."
        )
        ed.note(
            f"<b>Detalle técnico.</b> El VIF (factor de inflación de la varianza) máximo fue "
            f"{es(max_vif['VIF'], 2)} ({FEATURE_LABELS.get(max_vif['variable'], max_vif['variable'])}), por "
            "debajo del nivel habitual de alerta (5). La correlación entre variables es prácticamente nula "
            "en casi todos los pares."
        )

ed.subhead("¿Hay alguna variable que le esté «chivando» la respuesta al modelo?")
ed.passage(
    "Antes de entrenar un modelo hay que asegurarse de que no existe ninguna variable que revele "
    "prácticamente por sí sola cuál es el segmento del cliente. Eso sería <b>fuga de información</b>: "
    "el modelo parecería muy bueno, pero porque le estamos dando pistas que en una situación real no "
    "tendría.",
    tight=True,
)
ed.insight(
    "No encontramos ninguna variable con una separación casi perfecta. La variable más relacionada con "
    f"el segmento es {top_leak_name}, algo que tiene sentido: cuanto más vuela un cliente, más información "
    "aporta sobre su relación con la aerolínea. En resumen: hay <b>señal útil, pero no una respuesta "
    "escondida</b> en los datos.",
    aside=(
        f"Test ANOVA F / eta²: ninguna variable supera eta²=0,95 (separación casi perfecta); la más alta es "
        f"{top_leak_name} con eta²={es(top_leak['eta2'], 2)}."
    ),
)
with st.expander("Ver el test de fuga de información completo"):
    st.dataframe(
        leakage_df.rename(columns={"variable": "Variable", "F_stat": "F", "p_valor": "p-valor", "eta2": "eta²"}),
        use_container_width=True, hide_index=True,
    )

# ============================================================ 04 · EL MÉTODO ==
ed.beat(
    "metodo", "04", "Cómo se llegó al modelo", "Cómo construí una comparación justa",
    deck=(
        "No tiene sentido comparar tres modelos si cada uno recibe datos diferentes o se evalúa con reglas "
        "distintas. Por eso preparé <b>un mismo proceso para los tres</b>: mismos datos, mismo "
        "preprocesamiento, mismo entrenamiento y misma evaluación. La única diferencia entre ellos es la "
        "forma en la que cada algoritmo aprende de los datos."
    ),
)
ed.route(["Datos originales", "Revisión de calidad", "Límite a valores extremos (P1/P99)",
          "Completar vacíos y escalar", "Entrenamiento y prueba (80/20)", "3 modelos, mismo criterio"])
ed.steps([
    ("Partí de los datos ya tratados, no de los originales",
     "Los valores extremos de la sección anterior, limitados entre P1 y P99, son los que alimentan el "
     "modelo: así el tratamiento de outliers deja de ser un ejercicio aislado de exploración y pasa a "
     "<b>formar parte del proceso real</b>."),
    ("Dejé fuera una señal que un cliente nuevo no tendría",
     "El indicador de atipicidad de Isolation Forest se calculó sobre todo el dataset antes de separar "
     "entrenamiento y prueba: no es una variable de negocio disponible para un cliente nuevo. Usarla "
     "habría sido <b>hacer trampas</b>."),
    ("Completé los vacíos sin mirar la prueba final",
     "Nada se rellena a mano ni fuera del flujo: la mediana o la moda se calculan <b>solo con los datos de "
     "entrenamiento</b>, para que información de la prueba no se filtre al modelo (imputación y escalado "
     "dentro de un Pipeline de scikit-learn)."),
    ("Primero necesitaba saber qué significa hacerlo bien",
     f"Un clasificador que siempre predice el segmento más frecuente acierta el "
     f"<b>{pct(stats['baseline']['accuracy'], 0)}</b> de las veces. Ese es el suelo mínimo (baseline): ningún "
     "modelo real vale la pena si no lo supera con claridad."),
    ("Entrené los tres modelos en igualdad de condiciones",
     "Regresión Logística, Random Forest y Gradient Boosting, dentro del mismo Pipeline y sobre la misma "
     "partición de datos: la única diferencia entre ellos es <b>el algoritmo</b>, no la preparación."),
    ("Comparé los modelos antes de mirar la prueba final",
     "Una única partición 80/20 (300 clientes de prueba) tiene demasiada varianza para declarar un "
     "ganador. Por eso comparé primero los tres modelos con <b>validación cruzada</b> de 5 particiones "
     "estratificadas, usando solo los datos de entrenamiento."),
    ("La prueba final: datos que ningún modelo había visto",
     "Solo al final, reentrené cada modelo con todo el entrenamiento y lo evalué <b>una única vez</b> sobre "
     "el 20% de clientes que quedó completamente al margen: la prueba de que el ranking no es casualidad."),
])

ed.subhead("Tres formas diferentes de resolver el mismo problema")
ed.passage(
    "Probé tres algoritmos para responder exactamente a la misma pregunta: <b>¿a qué segmento pertenece "
    "este cliente?</b> Los tres reciben exactamente la misma información. Así podemos comparar sus "
    "resultados de forma justa.",
    tight=True,
)
with ed.figure("modelos"):
    ed.cols([
        {"tag": "Modelo lineal", "title": "Regresión Logística",
         "text": "Un modelo relativamente sencillo y fácil de interpretar. Es especialmente interesante "
                 "cuando queremos entender qué variables están detrás de cada predicción."},
        {"tag": "Árboles en paralelo", "title": "Random Forest",
         "text": "Combina muchos árboles de decisión para encontrar patrones más complejos en los datos."},
        {"tag": "Árboles en secuencia", "title": "Gradient Boosting",
         "text": "Construye los árboles de forma progresiva, utilizando los errores anteriores para mejorar "
                 "las siguientes predicciones."},
    ])

# ============================================================ 05 · LA EVIDENCIA ==
ed.beat("evidencia", "05", "La evidencia", "¿Funciona de verdad?", weight="major")

ed.subhead("¿Cuál funciona mejor?")
with ed.figure("cv", level="story"):
    st.plotly_chart(charts.cv_comparison(cv_df, stats["baseline"]["f1_macro"]),
                    use_container_width=True, config=PLOT)
    ed.caption("FIG. 03", "Accuracy y F1-macro medios de cada modelo en validación cruzada, con su "
                          "desviación. La línea roja marca el baseline.", FUENTE)
ed.insight(
    "Los tres modelos encuentran patrones útiles en el comportamiento de los clientes, pero "
    f"<b>{best_cv['modelo']} obtiene el mejor resultado</b> en la validación cruzada. Su F1-macro (una "
    f"medida que da el mismo peso a los tres segmentos) alcanza {es(best_cv['f1_macro_media'])}, frente a "
    f"{es(worst_cv['f1_macro_media'])} de {worst_cv['modelo']}. Pero todavía no podemos declararlo "
    "ganador: antes hay que comprobar que ese resultado se mantiene con clientes que nunca ha visto.",
    aside=(
        "Validación cruzada estratificada de 5 particiones, solo sobre entrenamiento. F1-macro medio "
        f"(±desviación): {best_cv['modelo']} {es(best_cv['f1_macro_media'])} (±{es(best_cv['f1_macro_std'])}) · "
        f"{worst_cv['modelo']} {es(worst_cv['f1_macro_media'])} (±{es(worst_cv['f1_macro_std'])}). Los tres "
        f"superan con claridad el baseline ({es(stats['baseline']['f1_macro'])}): hay señal real en los datos."
    ),
)

ed.subhead("¿Se mantiene con clientes que ningún modelo había visto?", level="wide")
with ed.split("prueba-final", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.test_comparison(test_df), use_container_width=True, config=PLOT)
        ed.caption("FIG. 04", "Accuracy y F1-macro de cada modelo sobre los 300 clientes de prueba.", FUENTE)
    with txt:
        ed.insight(
            "Sí: el ranking en la prueba final <b>coincide exactamente</b> con el de la validación cruzada. "
            "La señal es consistente, no un golpe de suerte de una única partición, y eso me da confianza "
            "para elegir el modelo ganador."
        )
        ed.note("<b>Detalle técnico.</b> Test hold-out de 300 clientes (el 20% reservado desde el principio), "
                "evaluado una única vez.")

ed.subhead("¿Dónde se equivoca?", level="wide")
with ed.figure("matrices"):
    for c, nombre in zip(st.columns(3), ["Regresión Logística", "Random Forest", "Gradient Boosting"]):
        with c:
            st.markdown(f'<p class="ed-metadata">{nombre}</p>', unsafe_allow_html=True)
            cm = confusion_matrices[nombre]
            st.plotly_chart(charts.confusion_matrix(cm["matrix"], cm["labels"], CLASS_LABELS),
                            use_container_width=True, config=PLOT)
    ed.caption("FIG. 05", "Matrices de confusión sobre la prueba final. Filas: segmento real. Columnas: "
                          "segmento predicho.", FUENTE)
ed.insight(
    f"{GANADOR} comete la mayoría de sus errores <b>entre segmentos vecinos</b>: Básico se confunde mucho "
    "más con Frecuente que con Premium. Y hay algo especialmente interesante: en el conjunto de prueba, "
    "<b>ningún cliente Premium fue clasificado como Básico</b>. El modelo no es perfecto, pero sus errores "
    "siguen un patrón lógico: le cuesta más distinguir a los clientes que están cerca del límite entre dos "
    "segmentos.",
    aside=(
        f"Matrices de confusión sobre el test. En {GANADOR}: Básico→Frecuente {gb_cm[0][1]} casos, "
        f"Básico→Premium {gb_cm[0][2]}, Premium→Básico {gb_cm[2][0]}."
    ),
)

ed.subhead("El resultado")
ed.insight(
    f"Después de probar los tres modelos, <b>{best_test['modelo']} obtiene el mejor rendimiento</b> tanto "
    "en validación cruzada como en el conjunto de prueba. En el test, clasifica correctamente alrededor "
    f"del <b>{pct(best_test['accuracy'], 0)} de los clientes</b>, frente al "
    f"{pct(stats['baseline']['accuracy'], 0)} que conseguiríamos simplemente asignando siempre el segmento "
    f"más frecuente. Y lo importante no es solo el {pct(best_test['accuracy'], 0)}: el ranking de los "
    "modelos se mantiene tanto en la validación como en la prueba final, lo que indica que el resultado es "
    "consistente y no depende de una partición concreta de los datos.",
    aside=(
        f"El {pct(best_test['accuracy'])} es la exactitud (accuracy) en test. En F1-macro, "
        f"{best_test['modelo']} logra {es(best_cv['f1_macro_media'])} en validación cruzada y "
        f"{es(best_test['f1_macro'])} en test, frente a {es(stats['baseline']['f1_macro'])} del baseline."
    ),
)

ed.subhead(f"¿Dónde acierta más y menos {GANADOR}?")
ed.metrics([
    (CLASS_LABELS[cls], es(report_gb[cls]["f1-score"], 2),
     f"F1. Cuando el modelo dice {CLASS_LABELS[cls]}, acierta el <b>{pct(report_gb[cls]['precision'], 0)}</b> "
     f"de las veces (precisión); y de todos los clientes {CLASS_LABELS[cls]} reales, encuentra el "
     f"<b>{pct(report_gb[cls]['recall'], 0)}</b> (recall).")
    for cls in ["Basico", "Frecuente", "Premium"]
])
ed.insight(
    f"El segmento más difícil de acertar es <b>Frecuente</b> (F1={es(report_gb['Frecuente']['f1-score'], 2)}), "
    "justo el que está en medio — comparte comportamiento con ambos vecinos. Básico y Premium, los dos "
    f"extremos, son más fáciles de distinguir (F1={es(report_gb['Basico']['f1-score'], 2)} y "
    f"{es(report_gb['Premium']['f1-score'], 2)}). Tiene sentido de negocio: el punto de corte entre "
    "\"empieza a viajar mucho\" y \"ya es Premium\" es, por naturaleza, más difuso que los extremos."
)
ed.band(
    "Lo que demuestra la evidencia",
    f'<span class="pos">{GANADOR}</span> acierta el segmento <span class="pos">{pct(best_test["accuracy"], 0)}</span> '
    f'de las veces — frente al <span class="neg">{pct(stats["baseline"]["accuracy"], 0)}</span> de asignar '
    "siempre el segmento más frecuente.",
    "Los tres modelos superan con claridad la referencia más simple: la señal es real. La elección final ya "
    "no es solo técnica — depende de lo que necesite el negocio.",
    quote=True,
)

# ============================================================ 06 · LO QUE HEMOS APRENDIDO ==
ed.beat(
    "aprendizajes", "06", "Explicabilidad", "No solo importa acertar. También importa entender por qué.",
    deck=(
        "Un modelo puede tener un buen rendimiento y, aun así, ser difícil de explicar. Por eso analizamos "
        "qué variables están más relacionadas con sus predicciones y qué podemos interpretar de cada "
        "algoritmo. El patrón es bastante claro: <b>la frecuencia de vuelo, la distancia recorrida y el "
        "gasto anual</b> aparecen entre las variables más relevantes, aunque cada modelo permite "
        "interpretar esa relación de una manera diferente."
    ),
)

ed.subhead("¿Qué variables pesan más en las decisiones de cada ensamble?", level="wide")
with ed.figure("importancias"):
    for c, (nombre, fi, color) in zip(
        st.columns(2, gap="large"),
        [("Random Forest", fi_rf, charts.NAVY3), (GANADOR, fi_gb, charts.INK)],
    ):
        with c:
            st.markdown(f'<p class="ed-metadata">{nombre} — variables más usadas</p>', unsafe_allow_html=True)
            st.plotly_chart(charts.feature_importance(fi, FEATURE_LABELS, color=color),
                            use_container_width=True, config=PLOT)
    ed.caption("FIG. 06", "Variables que más pesan en las decisiones de cada ensamble de árboles.", FUENTE)
ed.insight(
    "Los dos modelos coinciden en las variables más relevantes: frecuencia de viaje, distancia media y "
    "gasto anual. Importante: que una variable pese mucho no significa necesariamente que sea la causa de "
    "una decisión, y en este caso la importancia nativa puede favorecer determinadas variables numéricas. "
    "Por eso interpretamos estos resultados <b>con cautela</b>.",
    aside=(
        f"Las tres variables concentran más de la mitad de la importancia total ({pct(top3_share, 0)} o más en "
        "ambos modelos). <code>distancia_media_km</code> pesa mucho más aquí que en el test de fuga de "
        f"información (eta²={es(dist_eta2, 2)}): es un sesgo conocido de la importancia nativa, que tiende "
        "a sobrevalorar variables numéricas con muchos valores distintos frente a las categóricas o "
        "discretas."
    ),
)

ed.subhead("Regresión Logística — ¿qué variables empujan hacia Premium?", level="wide")
with ed.split("odds", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            "Aquí el signo importa: un <b>mayor gasto anual y una mayor frecuencia de viaje</b> están "
            "asociados a una mayor probabilidad de pertenecer al segmento Premium, manteniendo el resto de "
            "variables constantes, mientras que valores bajos empujan hacia Básico. Es la ventaja de un "
            "modelo lineal: el resultado se traduce a lenguaje de negocio sin intermediarios, algo que la "
            "importancia de variables de los ensambles no puede afirmar por sí sola."
        )
        ed.note(
            "<b>Detalle técnico.</b> Las variables numéricas se estandarizan antes de entrenar, así que cada "
            "odds ratio se lee por cada desviación estándar de aumento, no por euro ni por vuelo: para "
            f"Premium, {es(or_frec, 2)} en frecuencia de viaje y {es(or_gasto, 2)} en gasto anual. Son "
            "asociaciones del modelo, no efectos causales."
        )
    with viz:
        st.plotly_chart(charts.odds_ratios_class(odds_df, "Premium", FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 07", "Las ocho variables numéricas que más alejan la probabilidad de Premium (odds "
                              "ratio en escala log₂, por desviación estándar). Verde: más probable Premium; "
                              "rojo: menos.", FUENTE)

# ============================================================ 07 · PONLO A PRUEBA ==
ed.beat(
    "playground", "07", "Ponlo a prueba", "¿Cómo clasificarían los modelos a este cliente?", weight="major",
    deck=(
        "Ahora puedes probarlo tú mismo. <b>Modifica el comportamiento de un cliente</b> y observa cómo "
        "cambia la predicción de los tres modelos: cada uno muestra el segmento que predice y la "
        "probabilidad que asigna a cada opción. ¿Coinciden los tres? ¿Qué ocurre cuando el perfil del "
        "cliente es menos claro?"
    ),
)

num_defaults = playground_defaults["numeric"]
cat_options = playground_defaults["categorical_options"]
cat_defaults = playground_defaults["categorical_defaults"]
numeric_features = playground_defaults["numeric_features"]
categorical_features = playground_defaults["categorical_features"]

with ed.split("playground", "5-7") as (pg_left, pg_right):
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
        ed.metrics(
            [(nombre, CLASS_LABELS[preds[nombre]],
              f"{pct(results[nombre]['proba'][preds[nombre]])} de probabilidad", CLASS_COLORS[preds[nombre]])
             for nombre in ["Regresión Logística", "Random Forest", "Gradient Boosting"]],
            word=True,
        )
        st.plotly_chart(charts.playground_agreement(results, CLASS_LABELS), use_container_width=True, config=PLOT)
        ed.caption("FIG. 08", "Probabilidad que asigna cada modelo a cada segmento para el cliente definido "
                              "con los controles.", "Modelos entrenados en este proyecto")

ed.subhead("¿Están de acuerdo los tres modelos?")
if unanime:
    ed.insight(
        f"Los 3 modelos coinciden: <b>{CLASS_LABELS[list(preds.values())[0]]}</b>. Cuando el perfil del "
        "cliente es claro, hasta los modelos más simples llegan a la misma conclusión que el más complejo. "
        "La ventaja de un modelo sofisticado se nota sobre todo en <b>los casos ambiguos</b>, no en estos."
    )
else:
    ed.insight(
        "Los modelos <b>no están de acuerdo</b>: señal de que este cliente cae en una zona fronteriza "
        "entre dos segmentos. Es exactamente el tipo de caso donde la elección del modelo importa de "
        f"verdad, y donde {GANADOR} fue el que mejor generalizó en la prueba final."
    )

# ============================================================ 08 · IMPLICACIONES ==
ed.beat(
    "implicaciones", "08", "Del análisis a la acción", "¿Qué podría hacer una empresa con este análisis?",
    deck=(
        "Los resultados no dicen simplemente «utiliza este modelo»: permiten plantear <b>diferentes "
        "caminos</b> según la prioridad del negocio."
    ),
)
with ed.figure("prioridades"):
    ed.cols([
        {"title": "Si la prioridad es la explicabilidad",
         "text": "La Regresión Logística ofrece una forma más sencilla de entender qué variables están "
                 "asociadas a cada segmento."},
        {"title": "Si la prioridad es maximizar el rendimiento predictivo",
         "text": f"{GANADOR} obtiene el mejor resultado de los tres modelos evaluados."},
        {"title": "Si se busca una alternativa intermedia",
         "text": "Random Forest ofrece otra forma de capturar relaciones más complejas manteniendo una "
                 "interpretación relativamente accesible."},
    ])
ed.insight(
    "La elección final dependería del <b>contexto real</b>: coste de los errores, necesidad de "
    "explicabilidad, volumen de clientes y requisitos de negocio."
)

# ============================================================ 09 · LIMITACIONES ==
ed.beat("limitaciones", "09", "Honestidad ante todo", "Limitaciones")
ed.lists([
    {"title": "Lo que el modelo SÍ puede hacer", "points": [
        f"Clasificar <b>automáticamente</b> con {pct(best_test['accuracy'])} de acierto, muy por encima del "
        f"{pct(stats['baseline']['accuracy'], 0)} de la referencia más simple.",
        "Distinguir con claridad <b>los dos extremos</b> (Básico y Premium): casi nunca los confunde entre sí.",
        "Ofrecer <b>una versión explicable</b> (Regresión Logística) cuando Marketing necesita justificar la decisión.",
        "Señalar <b>qué variables pesan más</b>, abriendo la puerta a acciones comerciales dirigidas.",
    ]},
    {"title": "Lo que el modelo NO puede hacer", "points": [
        "Distinguir con la misma fiabilidad <b>al segmento intermedio</b> (Frecuente): es el que más se confunde.",
        "Compensar problemas en los <b>datos de entrada</b>.",
        "Mantenerse fiable si <b>el comportamiento de vuelo cambia</b> de forma estructural sin reentrenar.",
        "Generar <b>una decisión definitiva</b> por sí solo. El modelo proporciona una predicción que debe "
        "interpretarse dentro del contexto de negocio.",
    ]},
])
ed.passage(
    "El rendimiento depende por completo de la calidad de las variables de <b>incidencias y "
    "satisfacción</b>. Si su recogida en producción tiene sesgo o ruido (encuestas que solo responden los "
    "clientes más extremos, incidencias mal registradas), ningún modelo, por sofisticado que sea, lo "
    "compensa."
)

# ============================================================ 10 · CONCLUSIÓN ==
ed.beat(
    "conclusion", "10", "Conclusión", "Del dato a la decisión", weight="major",
    deck=[
        "El análisis demuestra que el comportamiento de los clientes contiene suficiente información para "
        "identificar automáticamente diferencias entre los segmentos Básico, Frecuente y Premium. De los "
        f"tres modelos evaluados, <b>{GANADOR} obtiene el mejor rendimiento</b> y mantiene ese resultado "
        "tanto en validación como en el conjunto de prueba.",
        "Pero el aprendizaje más importante no es simplemente qué modelo obtiene la mejor métrica. Es que "
        "el mismo problema puede tener <b>soluciones diferentes</b> dependiendo de lo que necesite el "
        "negocio: maximizar el rendimiento, entender las variables detrás de una predicción o encontrar un "
        "equilibrio entre ambas. Y ahí es donde los datos dejan de ser solo un conjunto de números y "
        "<b>empiezan a servir para tomar decisiones</b>.",
    ],
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/comparativa-modelos-aerolinea-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borja-mora-mendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
