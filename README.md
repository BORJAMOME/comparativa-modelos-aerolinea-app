# Comparativa de Modelos — Segmentación de Clientes de Aerolínea

**Tres modelos compiten por decidir en qué segmento cae cada cliente. Solo uno merece el puesto de verdad.**

Una aplicación interactiva que compara tres algoritmos de clasificación — Regresión Logística, Random
Forest y Gradient Boosting — entrenados con el mismo criterio para clasificar automáticamente a los
clientes de una aerolínea en tres niveles (Básico, Frecuente, Premium), y deja probar en vivo si los
tres modelos coinciden sobre un mismo cliente hipotético.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema, sigue por los
datos, compara los modelos con las mismas reglas, y termina dejándote construir un cliente hipotético
para ver qué opina cada uno.

## Ver la app

🔗 **[Abrir la app](https://comparativa-modelos-aerolinea.streamlit.app)** _(actualizar con la URL real tras el deploy en Streamlit Cloud)_

## De qué trata, en dos frases

Una aerolínea quiere clasificar automáticamente a sus clientes en tres segmentos para lanzar campañas
diferenciadas, pero no tiene un criterio automatizado. Se comparan tres modelos sobre 1.500 clientes y
20 variables de comportamiento, con el mismo split, el mismo preprocesamiento y la misma validación
cruzada, para que ganara el mejor modelo y no el más complejo por defecto.

**El resultado:** Gradient Boosting gana con un F1-macro de 0,784 en validación cruzada (0,799 en test),
un 18% mejor que la Regresión Logística y muy por encima del baseline (17,6%). Pero la app también
muestra por qué, en algunos casos, la Regresión Logística sigue siendo la opción correcta: cuando
Marketing necesita explicar la decisión, no solo acertarla.

## Qué te vas a encontrar al recorrerla

1. **El problema** — por qué una aerolínea necesita automatizar la segmentación de clientes
2. **Los datos** — 1.500 clientes, 21 variables originales, calidad del dato y outliers
3. **Antes de modelar** — outliers, multicolinealidad y fuga de información, antes de entrenar nada
4. **El camino hasta la comparativa** — cómo se garantiza que la comparación entre modelos sea justa
5. **Los 3 modelos cara a cara** — validación cruzada, test hold-out y matrices de confusión
6. **Explicabilidad** — qué variables pesan más en cada modelo, y por qué el signo importa en la logística
7. **Playground** — construye un cliente hipotético y mira si los 3 modelos están de acuerdo, en directo
8. **Resultados y decisiones** — qué modelo desplegar según lo que necesite el negocio

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y [scikit-learn](https://scikit-learn.org)
(`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`) para los modelos. El
análisis completo, en formato notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/clasificacion/04-comparativa-modelos/04-segmentacion-aerolinea).

Todos los números que aparecen en la app se calculan una vez en `model/train.py` y se guardan como
datos — nada está escrito a mano.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los resultados del modelo ya vienen calculados en `model/artifacts/`, así que no hace falta reentrenar
nada para verla funcionar.

Solo si cambias el dataset (`data/dataset_linea_aerea_multiclase_v2.xlsx`) necesitas regenerarlos:

```bash
python model/train.py    # tarda 1-2 minutos (validación cruzada de los 3 modelos)
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — toda la narrativa, sección a sección
components/
  ui.py                    bloques visuales reutilizables (tarjetas, títulos, callouts)
  charts.py                gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py            carga de artefactos y de los 3 modelos entrenados (con cache de Streamlit)
  classifier.py              predice el segmento de un cliente hipotético con los 3 modelos en vivo
model/
  train.py                    entrena y evalúa los 3 modelos, calcula todos los resultados
  artifacts/                   resultados ya calculados (métricas, matrices, los 3 pipelines entrenados)
data/                      el dataset original
assets/style.css           el sistema visual de la app
```

El Playground carga los 3 pipelines de scikit-learn ya entrenados (`model/artifacts/model_*.joblib` —
el mismo modelo, con los mismos pesos, que reporta las métricas de test) y llama a `predict_proba()` de
cada uno sobre el cliente hipotético, para poder comparar en vivo si los 3 modelos están de acuerdo.
</details>

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [GitHub](https://github.com/BORJAMOME)
