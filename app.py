"""
Comparativa de Modelos — Segmentación de Clientes de Aerolínea
Case study interactivo en Streamlit: tres algoritmos de clasificación
comparados de forma justa, del punto de partida a la decisión.

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


def es(value: float, decimals: int = 3) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value * 100, decimals)}%"


ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Clasificación de clientes</p>
      <h1 class="hero-title">¿Y si una aerolínea pudiera <em>reconocer</em> automáticamente qué tipo de cliente tiene delante?</h1>
      <p class="hero-sub">Un cliente que vuela una vez al año no debería recibir necesariamente la misma
      propuesta que uno que vuela todas las semanas. A partir de su comportamiento de vuelo, gasto y
      relación con la aerolínea, entrené y comparé tres modelos de Machine Learning para clasificar a
      cada cliente como Básico, Frecuente o Premium.</p>
      <p class="hero-sub">El objetivo: comprobar si los datos permiten automatizar esa clasificación y qué
      modelo lo hace mejor.</p>
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
    "La aerolínea tiene tres tipos de cliente: Básico, Frecuente y Premium. El reto es sencillo de "
    "plantear: si conocemos cómo vuela, cuánto gasta y cómo se relaciona con la compañía, ¿podemos "
    "utilizar esos datos para identificar automáticamente a qué segmento pertenece?"
)
ui.lead(
    "Esto permitiría pasar de campañas genéricas a acciones más adaptadas a cada tipo de cliente."
)
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Podemos <span class="accent">predecir el segmento de un cliente a partir de su comportamiento</span> '
    'y hacerlo con suficiente fiabilidad como para utilizarlo en campañas comerciales?',
    "Para responderla, comparé tres modelos diferentes utilizando exactamente los mismos datos y "
    "criterios de evaluación.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Punto de partida")
ui.h2("Los datos")
dist = stats["class_distribution"]
ui.lead(
    f"Partimos de {n_fmt} clientes y {stats['n_columns_original']} variables que describen diferentes "
    "aspectos de su relación con la aerolínea: frecuencia de vuelo, gasto, distancia recorrida, "
    "antigüedad, incidencias, satisfacción y fidelización."
)
ui.lead(
    "El objetivo que queremos predecir es el segmento del cliente: Básico, Frecuente o Premium. Es "
    "decir, el modelo aprende a reconocer patrones en el comportamiento de clientes que ya conocemos "
    "para poder aplicarlos después a nuevos clientes."
)
ui.tech_detail(
    "la variable objetivo, <code>segmento_cliente</code>, ya viene asignada por el negocio: el modelo "
    "aprende a reproducir ese criterio, no a inventar uno nuevo. Reparto de segmentos: "
    f"Básico {pct(dist['Basico'] / 100, 0)} · Frecuente {pct(dist['Frecuente'] / 100, 0)} · "
    f"Premium {pct(dist['Premium'] / 100, 0)}."
)

ui.h3("¿Qué información tenemos?")
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": f"{stats['n_columns_original']}", "label": "variables (incluido el segmento)"},
    {"num": "3", "label": "segmentos"},
    {"num": f"{len(nulos_df)}", "label": "variables con valores incompletos"},
])
ui.pipeline(["Datos originales", "Revisión de calidad", "Límite a valores extremos (P1/P99)",
             "Completar vacíos y escalar", "Entrenamiento y prueba (80/20)", "3 modelos, mismo criterio"])

ui.eyebrow("Calidad del dato", muted=True)
ui.h3("Antes de construir el modelo")
ui.lead(
    "Un modelo solo puede ser tan bueno como los datos con los que aprende. Antes de entrenar los "
    "modelos revisé tres aspectos: datos incompletos, valores extremos y variables que pudieran estar "
    "aportando información duplicada o demasiado cercana al resultado que queremos predecir."
)

ui.h3("¿Cuántos datos faltan?")
st.dataframe(
    nulos_df.rename(columns={"variable": "Variable", "nulos": "Nulos", "pct_nulos": "% nulos"}),
    use_container_width=True, hide_index=True,
)
ui.finding(
    f"Encontramos valores incompletos en {len(nulos_df)} variables. En lugar de eliminar esos clientes, "
    "completé los valores utilizando información disponible en el propio conjunto de entrenamiento. "
    f"Así mantenemos los {n_fmt} clientes y evitamos perder información innecesariamente.<br><br>"
    "El tratamiento de estos valores se realiza dentro del pipeline para evitar que información del "
    "conjunto de prueba influya en el entrenamiento."
)
ui.tech_detail(
    "entre el 1,5% y el 3,5% de nulos por variable: poco, pero suficiente para que un "
    "<code>dropna()</code> descartara el 10,8% de las filas. Imputación con la mediana (numéricas) y la "
    "moda (categóricas), ajustada solo sobre entrenamiento."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué me dicen los datos?")
ui.lead(
    "Resueltos los datos incompletos, quedan tres preguntas antes de entrenar nada: ¿hay clientes con "
    "comportamientos extremos? ¿Hay variables que cuentan prácticamente lo mismo? ¿Alguna variable le "
    "está «chivando» la respuesta al modelo?"
)

ui.h3("¿Hay clientes con comportamientos extremos?")
ui.body(
    "No todos los clientes tienen un comportamiento «normal»: algunos vuelan muchísimo más, gastan "
    "mucho más o recorren distancias muy superiores al resto. Estos valores pueden influir demasiado en "
    "un modelo, así que analizamos los posibles casos extremos antes de entrenarlo."
)
st.plotly_chart(charts.outlier_comparison(outlier_df, FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    f"En lugar de eliminar clientes, opté por limitar los valores más extremos entre los percentiles "
    f"P1 y P99. De esta forma conservamos los {n_fmt} clientes, pero evitamos que unos pocos valores "
    "extremos tengan un peso desproporcionado."
)
with st.expander("Detalle técnico: métodos de detección e Isolation Forest"):
    ui.body(
        "Los cuatro métodos no coinciden: es normal, cada uno mide algo distinto. Opté por "
        "<b>capar (winsorizing) en los percentiles P1/P99</b>, la opción más conservadora. Isolation "
        f"Forest, que sí mira todas las variables a la vez, señala un "
        f"{pct(stats['n_outliers_iforest']/stats['n_customers'])} de clientes como atípicos globales; "
        "lo calculé solo con fines de exploración, nunca lo usé como variable del modelo."
    )

ui.h3("¿Hay variables que cuentan prácticamente lo mismo?")
ui.body(
    "Si dos variables contienen información muy parecida, pueden aportar poco valor adicional al "
    "modelo. Por eso comprobé la relación entre las variables antes de entrenarlo."
)
st.plotly_chart(charts.correlation_heatmap(correlacion_df, FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
max_vif = vif_df.loc[vif_df["VIF"].idxmax()]
ui.finding(
    "El resultado fue positivo: no encontramos problemas relevantes de variables duplicadas o "
    "excesivamente relacionadas entre sí. Esto significa que podemos mantener la información "
    "disponible sin necesidad de eliminar variables por este motivo."
)
ui.tech_detail(
    f"el VIF (factor de inflación de la varianza) máximo fue {es(max_vif['VIF'], 2)} "
    f"({FEATURE_LABELS.get(max_vif['variable'], max_vif['variable'])}), por debajo del nivel habitual "
    "de alerta (5). La correlación entre variables es prácticamente nula en casi todos los pares."
)

ui.h3("¿Hay alguna variable que le esté «chivando» la respuesta al modelo?")
ui.body(
    "Antes de entrenar un modelo hay que asegurarse de que no existe ninguna variable que revele "
    "prácticamente por sí sola cuál es el segmento del cliente. Eso sería <b>fuga de información</b>: "
    "el modelo parecería muy bueno, pero porque le estamos dando pistas que en una situación real no "
    "tendría."
)
top_leak = leakage_df.iloc[0]
top_leak_name = FEATURE_LABELS.get(top_leak["variable"], top_leak["variable"])
ui.finding(
    "No encontramos ninguna variable con una separación casi perfecta. La variable más relacionada "
    f"con el segmento es <b>{top_leak_name}</b>, algo que tiene sentido: cuanto más vuela un cliente, "
    "más información aporta sobre su relación con la aerolínea.<br><br>"
    "<b>Conclusión:</b> encontramos señal útil, pero no una respuesta escondida en los datos."
)
ui.tech_detail(
    f"test ANOVA F / eta²: ninguna variable supera eta²=0,95 (separación casi perfecta); la más alta "
    f"es {top_leak_name} con eta²={es(top_leak['eta2'], 2)}."
)
with st.expander("Ver el test de fuga de información completo"):
    st.dataframe(
        leakage_df.rename(columns={"variable": "Variable", "F_stat": "F", "p_valor": "p-valor", "eta2": "eta²"}),
        use_container_width=True, hide_index=True,
    )
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo se llegó al modelo")
ui.h2("Cómo construí una comparación justa")
ui.lead(
    "No tiene sentido comparar tres modelos si cada uno recibe datos diferentes o se evalúa con reglas "
    "distintas. Por eso preparé un mismo proceso para los tres: mismos datos, mismo preprocesamiento, "
    "mismo entrenamiento y misma evaluación. La única diferencia entre ellos sería la forma en la que "
    "cada algoritmo aprende de los datos."
)
ui.story_steps([
    ("Partí de los datos ya tratados, no de los originales",
     "Los valores extremos de la sección anterior, limitados entre P1 y P99, son los que alimentan el "
     "modelo: así el tratamiento de outliers deja de ser un ejercicio aislado de exploración y pasa a "
     "formar parte del proceso real."),
    ("Dejé fuera una señal que un cliente nuevo no tendría",
     "El indicador de atipicidad de Isolation Forest se calculó sobre todo el dataset antes de separar "
     "entrenamiento y prueba: no es una variable de negocio disponible para un cliente nuevo. Usarla "
     "habría sido hacer trampas."),
    ("Completé los vacíos sin mirar la prueba final",
     "Nada se rellena a mano ni fuera del flujo: la mediana o la moda se calculan solo con los datos de "
     "entrenamiento, para que información de la prueba no se filtre al modelo (imputación y escalado "
     "dentro de un Pipeline de scikit-learn)."),
    ("Primero necesitaba saber qué significa hacerlo bien",
     f"Un clasificador que siempre predice el segmento más frecuente acierta el "
     f"{pct(stats['baseline']['accuracy'], 0)} de las veces. Ese es el suelo mínimo (baseline): ningún "
     "modelo real vale la pena si no lo supera con claridad."),
    ("Entrené los tres modelos en igualdad de condiciones",
     "Regresión Logística, Random Forest y Gradient Boosting, dentro del mismo Pipeline y sobre la misma "
     "partición de datos: la única diferencia entre ellos es el algoritmo, no la preparación."),
    ("Comparé los modelos antes de mirar la prueba final",
     "Una única partición 80/20 (300 clientes de prueba) tiene demasiada varianza para declarar un "
     "ganador. Por eso comparé primero los tres modelos con validación cruzada de 5 particiones "
     "estratificadas, usando solo los datos de entrenamiento."),
    ("La prueba final: datos que ningún modelo había visto",
     "Solo al final, reentrené cada modelo con todo el entrenamiento y lo evalué una única vez sobre el "
     "20% de clientes que quedó completamente al margen: la prueba de que el ranking no es casualidad."),
])
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Cómo intenta resolverlo?")
ui.h2("Tres formas diferentes de resolver el mismo problema")
ui.lead(
    "Probé tres algoritmos para responder exactamente a la misma pregunta: <b>¿a qué segmento pertenece "
    "este cliente?</b>"
)
mc1, mc2, mc3 = st.columns(3, gap="medium")
with mc1:
    ui.info_card(
        "Regresión Logística",
        "Un modelo relativamente sencillo y fácil de interpretar. Es especialmente interesante cuando "
        "queremos entender qué variables están detrás de cada predicción.",
        tag="Modelo lineal",
    )
with mc2:
    ui.info_card(
        "Random Forest",
        "Combina muchos árboles de decisión para encontrar patrones más complejos en los datos.",
        tag="Árboles en paralelo",
    )
with mc3:
    ui.info_card(
        "Gradient Boosting",
        "Construye los árboles de forma progresiva, utilizando los errores anteriores para mejorar las "
        "siguientes predicciones.",
        tag="Árboles en secuencia",
    )
st.write("")
ui.body(
    "Los tres reciben exactamente la misma información. Así podemos comparar sus resultados de forma "
    "justa."
)

best_cv = cv_df.iloc[0]
worst_cv = cv_df.iloc[-1]
ui.h3("¿Cuál funciona mejor?")
ui.finding(
    f"Los tres modelos encuentran patrones útiles en el comportamiento de los clientes, pero "
    f"<b>{best_cv['modelo']}</b> obtiene el mejor resultado en la validación cruzada. Su F1-macro "
    f"(una medida que da el mismo peso a los tres segmentos) alcanza {es(best_cv['f1_macro_media'])}, "
    f"frente a {es(worst_cv['f1_macro_media'])} de {worst_cv['modelo']}.<br><br>"
    "Pero todavía no podemos declararlo ganador. Antes necesitamos comprobar que ese resultado también "
    "se mantiene cuando el modelo se enfrenta a clientes que nunca ha visto."
)
st.plotly_chart(charts.cv_comparison(cv_df, stats["baseline"]["f1_macro"]), use_container_width=True,
                 config={"displayModeBar": False})
ui.tech_detail(
    "validación cruzada estratificada de 5 particiones, solo sobre entrenamiento. F1-macro medio "
    f"(±desviación): {best_cv['modelo']} {es(best_cv['f1_macro_media'])} (±{es(best_cv['f1_macro_std'])}) · "
    f"{worst_cv['modelo']} {es(worst_cv['f1_macro_media'])} (±{es(worst_cv['f1_macro_std'])}). Los tres "
    f"superan con claridad el baseline ({es(stats['baseline']['f1_macro'])}): hay señal real en los datos."
)

ui.h3("¿Se mantiene con clientes que ningún modelo había visto?")
ui.finding(
    "Sí: el ranking en la prueba final coincide exactamente con el de la validación cruzada. La señal "
    "es consistente, no un golpe de suerte de una única partición. Eso me da confianza para elegir el "
    "modelo ganador."
)
st.plotly_chart(charts.test_comparison(test_df), use_container_width=True, config={"displayModeBar": False})
ui.tech_detail("test hold-out de 300 clientes (el 20% reservado desde el principio), evaluado una única vez.")

gb_cm = confusion_matrices[GANADOR]["matrix"]
ui.h3("¿Dónde se equivoca?")
ui.finding(
    f"{GANADOR} comete la mayoría de sus errores entre segmentos vecinos. Por ejemplo, Básico se "
    "confunde mucho más con Frecuente que con Premium.<br><br>"
    "Y hay algo especialmente interesante: en el conjunto de prueba, <b>ningún cliente Premium fue "
    "clasificado como Básico</b>.<br><br>"
    "El modelo no es perfecto, pero sus errores siguen un patrón lógico: le cuesta más distinguir a los "
    "clientes que están cerca del límite entre dos segmentos."
)
cm_cols = st.columns(3)
for c, nombre in zip(cm_cols, ["Regresión Logística", "Random Forest", "Gradient Boosting"]):
    with c:
        st.markdown(f'<p class="co-body" style="font-weight:700; text-align:center;">{nombre}</p>',
                    unsafe_allow_html=True)
        cm = confusion_matrices[nombre]
        st.plotly_chart(charts.confusion_matrix(cm["matrix"], cm["labels"], CLASS_LABELS),
                         use_container_width=True, config={"displayModeBar": False})
ui.tech_detail(
    f"matrices de confusión sobre el test. En {GANADOR}: Básico→Frecuente {gb_cm[0][1]} casos, "
    f"Básico→Premium {gb_cm[0][2]}, Premium→Básico {gb_cm[2][0]}."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("Explicabilidad")
ui.h2("No solo importa acertar. También importa entender por qué.")
ui.lead(
    "Un modelo puede tener un buen rendimiento y, aun así, ser difícil de explicar. Por eso analizamos "
    "qué variables están más relacionadas con sus predicciones y qué podemos interpretar de cada "
    "algoritmo."
)
ui.lead(
    "Los resultados muestran un patrón bastante claro: la frecuencia de vuelo, la distancia recorrida y "
    "el gasto anual aparecen entre las variables más relevantes. Pero cada modelo nos permite "
    "interpretar esta relación de una manera diferente."
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
    "Los dos modelos coinciden en las variables más relevantes: frecuencia de viaje, distancia media y "
    "gasto anual.<br><br>"
    "<b>Importante:</b> la importancia de variables de estos modelos no significa necesariamente que "
    "una variable sea la causa de una decisión. En este caso, la importancia nativa puede favorecer "
    "determinadas variables numéricas. Por eso interpretamos estos resultados con cautela."
)
dist_eta2 = leakage_df.loc[leakage_df["variable"] == "distancia_media_km", "eta2"].iloc[0]
top3_share = min(fi_rf["importancia"].head(3).sum(), fi_gb["importancia"].head(3).sum())
ui.tech_detail(
    f"las tres variables concentran más de la mitad de la importancia total ({pct(top3_share, 0)} o más en "
    "ambos modelos). <code>distancia_media_km</code> pesa mucho más aquí que en el test de fuga de "
    f"información (eta²={es(dist_eta2, 2)}): es un sesgo conocido de la importancia nativa, que tiende "
    "a sobrevalorar variables numéricas con muchos valores distintos frente a las categóricas o "
    "discretas."
)

ui.h3("Regresión Logística — ¿qué variables empujan hacia Premium?")
st.plotly_chart(charts.odds_ratios_class(odds_df, "Premium", FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
odds_idx = odds_df.set_index("feature")
or_frec = odds_idx.loc["num__frecuencia_viaje_anual", "Premium"]
or_gasto = odds_idx.loc["num__gasto_anual_eur", "Premium"]
ui.finding(
    "Aquí el signo importa: un <b>mayor gasto anual</b> y una <b>mayor frecuencia de viaje</b> están "
    "asociados a una mayor probabilidad de pertenecer al segmento Premium, manteniendo el resto de "
    "variables constantes. Un gasto o una frecuencia bajos empujan, en cambio, hacia Básico.<br><br>"
    "Es la ventaja de un modelo lineal: el resultado se puede traducir a lenguaje de negocio sin "
    "intermediarios, algo que la importancia de variables de los ensambles no puede afirmar por sí sola."
)
ui.tech_detail(
    "las variables numéricas se estandarizan antes de entrenar, así que cada odds ratio se lee por "
    "cada desviación estándar de aumento, no por euro ni por vuelo: para Premium, "
    f"{es(or_frec, 2)} en frecuencia de viaje y {es(or_gasto, 2)} en gasto anual. Son asociaciones "
    "del modelo, no efectos causales."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Ponlo a prueba")
ui.h2("¿Cómo clasificarían los modelos a este cliente?")
ui.lead(
    "Ahora puedes probarlo tú mismo. Modifica el comportamiento de un cliente y observa cómo cambia la "
    "predicción de los tres modelos. Cada modelo muestra tanto el segmento que predice como la "
    "probabilidad que asigna a cada opción."
)
ui.lead("¿Coinciden los tres? ¿Qué ocurre cuando el perfil del cliente es menos claro?")

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
            ui.stat_card(nombre, CLASS_LABELS[pred], f"{pct(proba)} de probabilidad",
                         color=CLASS_COLORS[pred], value_size="1.1rem")
    st.write("")
    st.plotly_chart(charts.playground_agreement(results, CLASS_LABELS), use_container_width=True,
                     config={"displayModeBar": False})

ui.h3("¿Están de acuerdo los tres modelos?")
if unanime:
    ui.finding(
        f"Los 3 modelos coinciden: <b>{CLASS_LABELS[list(preds.values())[0]]}</b>. Cuando el perfil del "
        "cliente es claro, hasta los modelos más simples llegan a la misma conclusión que el más complejo. "
        "La ventaja de un modelo sofisticado se nota sobre todo en los casos ambiguos, no en estos."
    )
else:
    ui.finding(
        "Los modelos <b>no están de acuerdo</b>: señal de que este cliente cae en una zona fronteriza "
        "entre dos segmentos. Es exactamente el tipo de caso donde la elección del modelo importa de "
        f"verdad, y donde {GANADOR} fue el que mejor generalizó en la prueba final."
    )
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("¿Funciona de verdad?")
ui.h2("El resultado")
best_test = test_df.iloc[0]
worst_test = test_df.iloc[-1]
ui.lead(
    f"Después de probar los tres modelos, <b>{best_test['modelo']}</b> obtiene el mejor rendimiento tanto "
    "en validación cruzada como en el conjunto de prueba. En el test, clasifica correctamente alrededor "
    f"del <b>{pct(best_test['accuracy'], 0)}</b> de los clientes, frente al "
    f"<b>{pct(stats['baseline']['accuracy'], 0)}</b> que conseguiríamos simplemente asignando siempre el "
    "segmento más frecuente."
)
ui.lead(
    f"Lo importante no es solo el {pct(best_test['accuracy'], 0)}. El ranking de los modelos se mantiene "
    "tanto durante la validación como en la prueba final, lo que indica que el resultado es consistente "
    "y no depende únicamente de una partición concreta de los datos."
)
ui.tech_detail(
    f"el {pct(best_test['accuracy'])} es la exactitud (accuracy) en test. En F1-macro, {best_test['modelo']} "
    f"logra {es(best_cv['f1_macro_media'])} en validación cruzada y {es(best_test['f1_macro'])} en test, "
    f"frente a {es(stats['baseline']['f1_macro'])} del baseline."
)

report_gb = classification_reports[GANADOR]
ui.h3(f"¿Dónde acierta más y menos {GANADOR}?")
rep_cols = st.columns(3)
for c, cls in zip(rep_cols, ["Basico", "Frecuente", "Premium"]):
    with c:
        row = report_gb[cls]
        ui.stat_card(
            CLASS_LABELS[cls], es(row["f1-score"], 2),
            f'F1-score · precisión {es(row["precision"], 2)} · recall {es(row["recall"], 2)}',
            title_color=CLASS_COLORS[cls],
        )
st.write("")
ui.finding(
    f"El segmento más difícil de acertar es <b>Frecuente</b> (F1={es(report_gb['Frecuente']['f1-score'], 2)}), "
    "justo el que está en medio — comparte comportamiento con ambos vecinos. Básico y Premium, los dos "
    f"extremos, son más fáciles de distinguir (F1={es(report_gb['Basico']['f1-score'], 2)} y "
    f"{es(report_gb['Premium']['f1-score'], 2)}). Tiene sentido de negocio: el punto de corte entre "
    "\"empieza a viajar mucho\" y \"ya es Premium\" es, por naturaleza, más difuso que los extremos."
)
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'<span class="accent-pos">{GANADOR}</span> acierta el segmento '
    f'<span class="accent-pos">{pct(best_test["accuracy"], 0)}</span> de las veces — '
    f'frente al <span class="accent-neg">{pct(stats["baseline"]["accuracy"], 0)}</span> de asignar siempre el segmento más frecuente.',
    quote='"Los tres modelos superan con claridad la referencia más simple: la señal es real. La elección final ya no es solo técnica — depende de lo que necesite el negocio."',
)
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("Del análisis a la acción")
ui.h2("¿Qué podría hacer una empresa con este análisis?")
ui.lead(
    "Los resultados no dicen simplemente «utiliza este modelo». Permiten plantear diferentes caminos "
    "dependiendo de la prioridad del negocio."
)
dc1, dc2, dc3 = st.columns(3, gap="medium")
with dc1:
    ui.info_card(
        "Si la prioridad es la explicabilidad",
        "La Regresión Logística ofrece una forma más sencilla de entender qué variables están asociadas "
        "a cada segmento.",
    )
with dc2:
    ui.info_card(
        "Si la prioridad es maximizar el rendimiento predictivo",
        f"{GANADOR} obtiene el mejor resultado de los tres modelos evaluados.",
    )
with dc3:
    ui.info_card(
        "Si se busca una alternativa intermedia",
        "Random Forest ofrece otra forma de capturar relaciones más complejas manteniendo una "
        "interpretación relativamente accesible.",
    )
st.write("")
ui.finding(
    "La elección final dependería del contexto real: coste de los errores, necesidad de explicabilidad, "
    "volumen de clientes y requisitos de negocio."
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
        <li>Clasificar automáticamente y con {pct(best_test['accuracy'])} de acierto, muy por encima del {pct(stats['baseline']['accuracy'], 0)} de la referencia más simple.</li>
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
        <li>Compensar problemas en los datos de entrada.</li>
        <li>Mantenerse fiable si el comportamiento de vuelo cambia de forma estructural sin reentrenar.</li>
        <li>Generar una decisión definitiva por sí solo. El modelo proporciona una predicción que debe interpretarse dentro del contexto de negocio.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "El rendimiento depende por completo de la calidad de las variables de incidencias y satisfacción. "
    "Si su recogida en producción tiene sesgo o ruido (encuestas que solo responden los clientes más "
    "extremos, incidencias mal registradas), ningún modelo, por sofisticado que sea, lo compensa."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Conclusión")
ui.h2("Del dato a la decisión")
ui.lead(
    "El análisis demuestra que el comportamiento de los clientes contiene suficiente información para "
    "identificar automáticamente diferencias entre los segmentos Básico, Frecuente y Premium. De los "
    f"tres modelos evaluados, <b>{GANADOR}</b> obtiene el mejor rendimiento y mantiene ese resultado "
    "tanto en validación como en el conjunto de prueba."
)
ui.lead(
    "Pero el aprendizaje más importante no es simplemente qué modelo obtiene la mejor métrica. Es que el "
    "mismo problema puede tener soluciones diferentes dependiendo de lo que necesite el negocio: "
    "maximizar el rendimiento, entender las variables detrás de una predicción o encontrar un equilibrio "
    "entre ambas."
)
ui.lead(
    "Y ahí es donde los datos dejan de ser solo un conjunto de números y empiezan a servir para tomar "
    "decisiones."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/comparativa-modelos-aerolinea-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
