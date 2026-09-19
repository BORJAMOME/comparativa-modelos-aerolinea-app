"""
Comparativa de Modelos — Segmentación de Clientes de Aerolínea
Reportaje digital de datos: tres algoritmos de clasificación comparados de forma justa,
del problema de negocio a la decisión.

Narrativa (13 beats):
  LEDE → 01 El problema → 02 Los datos → 03 Antes de construir el modelo →
  04 Cómo construí una comparación justa → 05 Tres formas de resolver el problema →
  06 ¿Cuál funciona mejor? → 07 ¿Dónde se equivoca? → 08 No solo importa acertar →
  09 Ponlo a prueba → 10 El resultado → 11 ¿Qué podría hacer una empresa? →
  12 Limitaciones → 13 Del dato a la decisión

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial
reutilizable); aquí solo hay contenido y datos. Toda cifra del texto sale de los artefactos
de model/artifacts (generados por model/train.py): nada está escrito a mano.

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
LECTURA = "15 min"
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
n_train, n_test = stats["n_train"], stats["n_test"]
n_train_fmt = f"{n_train:,}".replace(",", ".")
GANADOR = stats["modelo_ganador"]          # el de mayor F1-macro en la prueba final (train.py)
dist = stats["class_distribution"]
base_acc, base_f1 = stats["baseline"]["accuracy"], stats["baseline"]["f1_macro"]

# Ordenados de mejor a peor F1-macro (validación cruzada / prueba final)
best_cv, second_cv, worst_cv = cv_df.iloc[0], cv_df.iloc[1], cv_df.iloc[2]
best_test, second_test, worst_test = test_df.iloc[0], test_df.iloc[1], test_df.iloc[2]
ok = {r["modelo"]: round(r["accuracy"] * n_test) for _, r in test_df.iterrows()}   # clientes bien clasificados
report_gb = classification_reports[GANADOR]
gb_cm = confusion_matrices[GANADOR]["matrix"]
gb_errores = sum(map(sum, gb_cm)) - sum(gb_cm[i][i] for i in range(3))
gb_errores_vecinos = sum(gb_cm[i][j] for i in range(3) for j in range(3) if abs(i - j) == 1)
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
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("validacion", "Validación"),
     ("explicabilidad", "Explicabilidad"), ("playground", "Playground"), ("resultado", "Resultado"),
     ("implicaciones", "Implicaciones"), ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Clasificación de clientes",
    headline="¿Y si una aerolínea pudiera <em>reconocer</em> automáticamente qué tipo de cliente tiene delante?",
    deck=(
        "Un cliente que vuela una vez al año no debería recibir necesariamente la misma propuesta que uno "
        "que vuela todas las semanas. A partir de su comportamiento de vuelo, gasto y relación con la "
        "aerolínea, se entrenaron y compararon tres modelos de Machine Learning para clasificar a cada "
        "cliente como <b>Básico, Frecuente o Premium</b>. El objetivo: comprobar si los datos permiten "
        "automatizar esa clasificación y <b>qué modelo lo hace mejor</b>."
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
    "contexto", "01", "Problema", "El problema",
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
    "Para responderla, se compararon tres modelos diferentes utilizando exactamente los mismos datos y "
    "criterios de evaluación.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
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
    "El segmento, <code>segmento_cliente</code>, ya viene en los datos: el modelo aprende a reproducir esa "
    "clasificación existente, no a inventar una nueva."
)

# ============================================================ 03 · ANTES DE CONSTRUIR EL MODELO ==
ed.beat(
    "calidad", "03", "Calidad del dato", "Antes de construir el modelo",
    deck=(
        "Un modelo solo puede ser tan bueno como los datos con los que aprende. Antes de entrenar los "
        "modelos se revisaron tres aspectos: <b>datos incompletos</b>, <b>valores extremos</b> y variables "
        "que aporten <b>información duplicada</b> o demasiado cercana al resultado que queremos predecir."
    ),
)

ed.subhead("¿Cuántos datos faltan?", level="wide")
with ed.split("nulos", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Encontramos valores incompletos en {len(nulos_df)} variables. En lugar de eliminar esos "
            "clientes, se completaron los valores utilizando información disponible en el propio conjunto "
            f"de entrenamiento. Así <b>mantenemos los {n_fmt} clientes</b> y evitamos perder información "
            "innecesariamente. Este paso se hace dentro del propio proceso de entrenamiento, para evitar que "
            "información de la prueba final influya en el modelo."
        )
        ed.note(
            "<b>Detalle técnico.</b> Entre el 1,5% y el 3,5% de nulos por variable: poco, pero suficiente "
            "para que un <code>dropna()</code> descartara el 10,8% de las filas. Imputación con la mediana "
            "(numéricas) y la moda (categóricas) dentro de un <code>Pipeline</code> de scikit-learn, "
            "ajustada solo sobre entrenamiento."
        )
    with viz:
        st.dataframe(
            nulos_df.rename(columns={"variable": "Variable", "nulos": "Nulos", "pct_nulos": "% nulos"}),
            use_container_width=True, hide_index=True,
        )
        ed.caption("TAB. 01", "Variables con valores incompletos: número y porcentaje de nulos.", FUENTE)

ed.subhead("¿Hay clientes con comportamientos extremos?", level="wide")
with ed.split("extremos", "5-7") as (txt, viz):
    with txt:
        ed.passage(
            "No todos los clientes tienen un comportamiento «normal»: algunos vuelan muchísimo más, gastan "
            "mucho más o recorren distancias muy superiores al resto. Estos valores pueden <b>influir "
            "demasiado</b> en un modelo, así que se analizaron los posibles casos extremos antes de entrenarlo."
        )
        ed.insight(
            "En lugar de eliminar clientes, se optó por <b>limitar los valores más extremos</b> entre los "
            f"percentiles P1 y P99. De esta forma se conservan los {n_fmt} clientes, pero se evita que unos "
            "pocos valores extremos tengan un peso desproporcionado."
        )
        ed.note(
            "<b>Detalle técnico.</b> Los tres métodos comparados no coinciden: es normal, cada uno mide algo "
            "distinto. Se eligió el capado (winsorizing) en P1/P99 por ser el más conservador. Los "
            "percentiles se calcularon con los "
            f"{n_fmt} clientes, antes de separar entrenamiento y prueba (ver Limitaciones). Isolation Forest, "
            f"configurado para marcar el 3% más atípico, señala {stats['n_outliers_iforest']} clientes como "
            "atípicos globales; se calculó solo con fines de exploración y no se usó como variable del modelo."
        )
    with viz:
        st.plotly_chart(charts.outlier_comparison(outlier_df, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Las ocho variables con más valores atípicos según el método IQR, comparadas con "
                              "los resultados de Z-score y de los percentiles P1/P99.", FUENTE)

ed.subhead("¿Hay variables que cuentan prácticamente lo mismo?", level="wide")
with ed.split("duplicadas", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.correlation_heatmap(correlacion_df, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 02", "Correlación entre pares de variables numéricas.", FUENTE)
    with txt:
        ed.passage(
            "Si dos variables contienen información muy parecida, pueden aportar <b>poco valor adicional</b> "
            "al modelo. Por eso se comprobó la relación entre las variables antes de entrenarlo."
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

# ============================================================ 04 · CÓMO CONSTRUÍ UNA COMPARACIÓN JUSTA ==
ed.beat(
    "metodo", "04", "Método", "Cómo construí una comparación justa",
    deck=(
        "No tiene sentido comparar tres modelos si cada uno recibe datos diferentes o se evalúa con reglas "
        "distintas. Por eso se preparó <b>un mismo proceso para los tres</b>: mismos datos, mismo "
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
     "<b>formar parte del proceso real</b>. Los percentiles se calcularon antes de separar entrenamiento "
     "y prueba (ver Limitaciones)."),
    ("Dejé fuera una señal que un cliente nuevo no tendría",
     "El indicador de atipicidad de Isolation Forest se calculó sobre todo el dataset antes de separar "
     "entrenamiento y prueba: no es una variable de negocio disponible para un cliente nuevo. Usarla "
     "habría sido <b>hacer trampas</b>."),
    ("Completé los vacíos sin mirar la prueba final",
     "Nada se rellena a mano ni fuera del flujo: la mediana o la moda se calculan <b>solo con los datos de "
     "entrenamiento</b>, para que información de la prueba no se filtre al modelo (imputación y escalado "
     "dentro del proceso de entrenamiento, con un Pipeline de scikit-learn)."),
    ("Primero necesitaba saber qué significa hacerlo bien",
     f"Un clasificador que siempre predice el segmento más frecuente acierta el "
     f"<b>{pct(base_acc, 0)}</b> de las veces. Ese es el suelo mínimo (baseline): ningún "
     "modelo real vale la pena si no lo supera con claridad."),
    ("Entrené los tres modelos en igualdad de condiciones",
     "Regresión Logística, Random Forest y Gradient Boosting, dentro del mismo proceso de preparación y "
     "entrenamiento y sobre la misma partición de datos: la única diferencia entre ellos es "
     "<b>el algoritmo</b>, no la preparación."),
    ("Comparé los modelos antes de mirar la prueba final",
     f"Una única partición 80/20 ({n_test} clientes de prueba) tiene demasiada varianza para sacar "
     "conclusiones. Por eso se comparó primero con <b>validación cruzada</b> de 5 particiones "
     f"estratificadas, usando solo los {n_train_fmt} clientes de entrenamiento."),
    ("La prueba final: datos que ningún modelo había visto",
     f"Solo al final, cada modelo se reentrenó con todo el entrenamiento y se evaluó <b>una única vez</b> "
     f"sobre el 20% de clientes ({n_test}) que quedó completamente al margen: una segunda comprobación, "
     "independiente de la primera, de que el ranking se mantiene."),
])

# ============================================================ 05 · TRES FORMAS DE RESOLVER EL PROBLEMA ==
ed.beat(
    "modelos", "05", "Modelos", "Tres formas de resolver el problema", weight="minor",
    deck=(
        "Se probaron tres algoritmos para responder exactamente a la misma pregunta: <b>¿a qué segmento "
        "pertenece este cliente?</b> Los tres reciben exactamente la misma información. Así podemos "
        "comparar sus resultados de forma justa."
    ),
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

# ============================================================ 06 · ¿CUÁL FUNCIONA MEJOR? ==
ed.beat(
    "validacion", "06", "Validación", "¿Cuál funciona mejor?", weight="major",
    deck=(
        "La comparación se hace en dos pasos: primero con <b>validación cruzada</b> sobre los datos de "
        "entrenamiento y después, una única vez, con clientes que ningún modelo ha visto."
    ),
)
with ed.figure("cv", level="story"):
    st.plotly_chart(charts.cv_comparison(cv_df, base_f1), use_container_width=True, config=PLOT)
    ed.caption("FIG. 03", "Accuracy y F1-macro medios de cada modelo en validación cruzada, con su "
                          "desviación. La línea roja marca el baseline.", FUENTE)
ed.insight(
    "Los tres modelos encuentran patrones útiles en el comportamiento de los clientes, pero "
    f"<b>{best_cv['modelo']} obtiene el mejor resultado</b> en la validación cruzada. Su F1-macro (una "
    f"medida que da el mismo peso a los tres segmentos) alcanza {es(best_cv['f1_macro_media'])}, frente a "
    f"{es(worst_cv['f1_macro_media'])} de {worst_cv['modelo']}. Esto se ha medido solo con los datos de "
    "entrenamiento: antes de darlo por bueno hay que ver si el resultado se mantiene con clientes que "
    "nunca ha visto.",
    aside=(
        "Validación cruzada estratificada de 5 particiones, solo sobre entrenamiento. F1-macro medio "
        f"(±desviación): {best_cv['modelo']} {es(best_cv['f1_macro_media'])} (±{es(best_cv['f1_macro_std'])}) · "
        f"{second_cv['modelo']} {es(second_cv['f1_macro_media'])} (±{es(second_cv['f1_macro_std'])}) · "
        f"{worst_cv['modelo']} {es(worst_cv['f1_macro_media'])} (±{es(worst_cv['f1_macro_std'])}). Los tres "
        f"superan con claridad el baseline ({es(base_f1)} de F1-macro)."
    ),
)

ed.subhead("¿Se mantiene con clientes que ningún modelo había visto?", level="wide")
with ed.split("prueba-final", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.test_comparison(test_df), use_container_width=True, config=PLOT)
        ed.caption("FIG. 04", f"Accuracy y F1-macro de cada modelo sobre los {n_test} clientes de prueba.", FUENTE)
    with txt:
        ed.insight(
            "Sí: el ranking en la prueba final <b>coincide exactamente</b> con el de la validación cruzada. "
            "Que el orden se repita en dos evaluaciones distintas da más confianza en que la ventaja de los "
            "ensambles de árboles sobre la Regresión Logística no es casual."
        )
        ed.note(f"<b>Detalle técnico.</b> Test hold-out de {n_test} clientes (el 20% reservado desde el "
                "principio), evaluado una única vez.")

# ============================================================ 07 · ¿DÓNDE SE EQUIVOCA? ==
ed.beat("errores", "07", "Evidencia", "¿Dónde se equivoca?", weight="minor")
with ed.figure("matrices", level="full"):
    for c, nombre in zip(st.columns(3, gap="large"), ["Regresión Logística", "Random Forest", "Gradient Boosting"]):
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
        f"{gb_errores_vecinos} de los {gb_errores} errores de {GANADOR} ocurren entre segmentos vecinos. "
        f"Básico→Frecuente {gb_cm[0][1]} casos, Básico→Premium {gb_cm[0][2]}, Premium→Básico {gb_cm[2][0]}."
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

# ============================================================ 08 · NO SOLO IMPORTA ACERTAR ==
ed.beat(
    "explicabilidad", "08", "Explicabilidad", "No solo importa acertar. También importa entender por qué.",
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

ed.subhead("Regresión Logística — ¿qué variables se asocian con Premium?", level="wide")
with ed.split("odds", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            "Aquí el signo importa: un <b>mayor gasto anual y una mayor frecuencia de viaje</b> están "
            "asociados a una mayor probabilidad de pertenecer al segmento Premium, manteniendo constantes el "
            "resto de variables, mientras que valores bajos se asocian con una mayor probabilidad de Básico. "
            "Es la ventaja de un modelo lineal: el resultado se traduce a lenguaje de negocio sin "
            "intermediarios, algo que la importancia de variables de los ensambles no puede afirmar por sí sola."
        )
        ed.note(
            "<b>Detalle técnico.</b> Cada valor es exp(coeficiente) de la regresión logística multinomial. "
            "Las variables numéricas se estandarizan antes de entrenar, así que se lee por cada desviación "
            "estándar de aumento (no por euro ni por vuelo) y describe una asociación relativa al resto de "
            f"segmentos, no un efecto causal. Para Premium: {es(or_frec, 2)} en frecuencia de viaje y "
            f"{es(or_gasto, 2)} en gasto anual."
        )
    with viz:
        st.plotly_chart(charts.odds_ratios_class(odds_df, "Premium", FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 07", "Las ocho variables numéricas más asociadas con un cambio en la probabilidad de "
                              "Premium (exp(coeficiente) en escala log₂, por desviación estándar). Verde: más "
                              "probable Premium; rojo: menos.", FUENTE)

# ============================================================ 09 · PONLO A PRUEBA ==
ed.beat(
    "playground", "09", "Playground", "Ponlo a prueba", weight="major",
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
        f"verdad, y donde {GANADOR} obtuvo el mejor resultado en la prueba final."
    )

# ============================================================ 10 · EL RESULTADO ==
ed.beat("resultado", "10", "Resultado", "El resultado", weight="major")
ed.band(
    "La conclusión",
    f'<span class="pos">{GANADOR}</span> obtiene el mejor rendimiento según F1-macro y accuracy, tanto en '
    "validación cruzada como en la prueba final.",
    f"{second_test['modelo']} queda muy cerca, y los tres modelos superan con claridad la referencia más "
    "simple: la señal es real. La elección final ya no es solo técnica — depende de lo que necesite el negocio.",
    quote=True,
)
ed.metrics([
    ("Accuracy · prueba final", pct(best_test["accuracy"]),
     f"{GANADOR} clasifica correctamente <b>{ok[GANADOR]} de los {n_test}</b> clientes de la prueba final. "
     f"El baseline, asignando siempre el segmento más frecuente, acertaría el {pct(base_acc, 0)}."),
    ("F1-macro · prueba final", es(best_test["f1_macro"]),
     f"Promedia el acierto en Básico, Frecuente y Premium dando el mismo peso a cada segmento. El baseline "
     f"obtiene {es(base_f1)}."),
    ("F1-macro · validación", es(best_cv["f1_macro_media"]),
     f"Con los datos de entrenamiento y una variación de ±{es(best_cv['f1_macro_std'])} entre particiones: "
     "coherente con lo que se ve en la prueba final."),
])
ed.insight(
    f"{second_test['modelo']} queda muy cerca: clasifica correctamente {ok[second_test['modelo']]} de los "
    f"{n_test} clientes de prueba, frente a {ok[GANADOR]} de {GANADOR}. En validación cruzada la diferencia "
    f"entre ambos ({es(best_cv['f1_macro_media'] - second_cv['f1_macro_media'])} de F1-macro) es menor que la "
    f"variación entre particiones (±{es(best_cv['f1_macro_std'])}). {worst_test['modelo']} queda más atrás: "
    f"{ok[worst_test['modelo']]} de {n_test}.",
    aside=(
        f"F1-macro en la prueba final: {GANADOR} {es(best_test['f1_macro'])} · "
        f"{second_test['modelo']} {es(second_test['f1_macro'])} · "
        f"{worst_test['modelo']} {es(worst_test['f1_macro'])}. En validación cruzada: "
        f"{es(best_cv['f1_macro_media'])} · {es(second_cv['f1_macro_media'])} · {es(worst_cv['f1_macro_media'])}."
    ),
)
ed.passage(
    "Qué significa: los tres modelos <b>superan con claridad al baseline</b> y los dos ensambles de árboles "
    f"superan claramente a la Regresión Logística. Entre {GANADOR} y {second_test['modelo']}, con estos datos "
    "no se puede afirmar con seguridad cuál es mejor.",
    tight=True,
)

# ============================================================ 11 · ¿QUÉ PODRÍA HACER UNA EMPRESA? ==
ed.beat(
    "implicaciones", "11", "Implicaciones", "¿Qué podría hacer una empresa con este análisis?",
    deck=(
        "Los resultados no dicen simplemente «utiliza este modelo»: permiten plantear <b>diferentes "
        "caminos</b> según la prioridad del negocio."
    ),
)
with ed.figure("prioridades"):
    ed.cols([
        {"tag": "Explicabilidad", "title": "Si la prioridad es la explicabilidad",
         "text": "La Regresión Logística ofrece una forma más sencilla de entender qué variables están "
                 "asociadas a cada segmento."},
        {"tag": "Rendimiento", "title": "Si la prioridad es maximizar el rendimiento predictivo",
         "text": f"{GANADOR} obtiene el mejor resultado de los tres modelos evaluados, aunque por poco "
                 f"frente a {second_test['modelo']}."},
        {"tag": "Equilibrio", "title": "Si se busca una alternativa intermedia",
         "text": "Random Forest ofrece otra forma de capturar relaciones más complejas manteniendo una "
                 "interpretación relativamente accesible."},
    ])
ed.insight(
    "La elección final dependería del <b>contexto real</b>: coste de los errores, necesidad de "
    "explicabilidad, volumen de clientes y requisitos de negocio."
)

# ============================================================ 12 · LIMITACIONES ==
ed.beat("limitaciones", "12", "Limitaciones", "Limitaciones")
ed.lists([
    {"title": "Lo que el modelo SÍ puede hacer", "points": [
        f"Clasificar <b>automáticamente</b> con {pct(best_test['accuracy'])} de clasificaciones correctas en la "
        f"prueba final, muy por encima del {pct(base_acc, 0)} de la referencia más simple.",
        "Distinguir con claridad <b>los dos extremos</b> (Básico y Premium): casi nunca los confunde entre sí.",
        "Ofrecer <b>una versión explicable</b> (Regresión Logística) cuando Marketing necesita justificar la decisión.",
        "Señalar <b>qué variables pesan más</b>, abriendo la puerta a acciones comerciales dirigidas.",
    ]},
    {"title": "Lo que el modelo NO puede hacer", "points": [
        "Distinguir con la misma fiabilidad <b>al segmento intermedio</b> (Frecuente): es el que más se confunde.",
        "Compensar problemas en los <b>datos de entrada</b>.",
        "Mantenerse fiable si <b>el comportamiento de vuelo cambia</b> de forma estructural sin reentrenar.",
        "Demostrar <b>causalidad</b>: las variables se asocian con el segmento, pero este análisis no prueba "
        "que lo provoquen.",
        f"Garantizar el mismo rendimiento <b>fuera de este conjunto de datos</b>: se evaluó con {n_fmt} "
        "clientes de un único conjunto.",
        "Generar <b>una decisión definitiva</b> por sí solo. El modelo proporciona una predicción que debe "
        "interpretarse dentro del contexto de negocio.",
    ]},
])
ed.passage(
    "El rendimiento depende por completo de la calidad de las variables de <b>incidencias y "
    "satisfacción</b>. Si su recogida en producción tiene sesgo o ruido (encuestas que solo responden los "
    "clientes más extremos, incidencias mal registradas), ningún modelo, por sofisticado que sea, lo "
    "compensa.",
    "Antes de un uso real haría falta una <b>validación adicional</b>, con datos nuevos y en el contexto de "
    "producción.",
    aside=(
        f"Los percentiles P1/P99 del capado se calcularon con los {n_fmt} clientes antes de separar "
        "entrenamiento y prueba (la imputación y el escalado sí se ajustan solo con el entrenamiento); no se "
        "ha medido cuánto influye en los resultados. Además, los tres modelos se entrenaron con "
        "hiperparámetros fijos, sin ajuste."
    ),
)

# ============================================================ 13 · DEL DATO A LA DECISIÓN ==
ed.beat(
    "conclusion", "13", "Conclusión", "Del dato a la decisión", weight="major",
    deck=[
        f"Los datos de comportamiento de {n_fmt} clientes contienen información suficiente para distinguir "
        "entre los segmentos Básico, Frecuente y Premium: los tres modelos superan con claridad la referencia "
        f"más simple, y el mejor, <b>{GANADOR}</b>, clasifica correctamente el {pct(best_test['accuracy'], 0)} "
        f"de los clientes de la prueba final frente al {pct(base_acc, 0)}.",
        "Pero el aprendizaje más importante no es qué modelo obtiene la mejor métrica —la diferencia entre "
        f"{GANADOR} y {second_test['modelo']} es pequeña—, sino que "
        "el mismo problema puede tener <b>soluciones diferentes</b> dependiendo de lo que necesite el "
        "negocio: maximizar el rendimiento, entender qué variables se asocian con cada segmento o buscar un "
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
