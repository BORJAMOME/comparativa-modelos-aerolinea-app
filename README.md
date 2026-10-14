# Comparativa de Modelos - Segmentación de Clientes de Aerolínea

En este proyecto he comparado tres modelos de Machine Learning para clasificar a los clientes de una aerolínea en tres segmentos: Básico, Frecuente y Premium.

He trabajado con los mismos datos y criterios de evaluación para comparar la Regresión Logística, Random Forest y Gradient Boosting. Además, he preparado una aplicación en la que se puede probar un cliente hipotético y comprobar qué segmento le asigna cada modelo.

## Ver la aplicación

**[Abrir la aplicación interactiva](https://comparativa-modelos-aerolinea.streamlit.app)**

## El análisis

He trabajado con un conjunto de datos de 1.500 clientes y 21 variables originales. Antes de entrenar los modelos, he revisado la calidad de los datos, los valores atípicos, la multicolinealidad y posibles problemas de fuga de información.

Después, he comparado los tres algoritmos utilizando el mismo preprocesamiento, la misma partición de los datos y validación cruzada. Así he podido evaluar sus resultados bajo las mismas condiciones.

### Resultados

Gradient Boosting ha obtenido el mejor resultado, con un F1-macro de 0,784 en validación cruzada y de 0,799 en el conjunto de test.

En esta comparación, su resultado en validación cruzada es aproximadamente un 18 % superior al de la Regresión Logística. También he incluido un modelo de referencia (*baseline*) para comprobar cuánto mejoran los algoritmos respecto a una predicción sencilla.

Sin embargo, el resultado no es el único factor que he tenido en cuenta. La Regresión Logística permite interpretar con más facilidad cómo influyen las variables en las predicciones, algo que puede ser relevante cuando también importa explicar por qué se asigna un segmento a un cliente.

## Qué encontrarás en la aplicación

He organizado el análisis en ocho apartados:

1. **El problema:** por qué puede ser útil clasificar a los clientes en distintos segmentos.
2. **Los datos:** qué información contiene el conjunto de datos y qué problemas he encontrado.
3. **Preparación de los datos:** revisión de valores atípicos, multicolinealidad y posibles fugas de información.
4. **Preparación de los modelos:** cómo he organizado el proceso para poder compararlos en las mismas condiciones.
5. **Comparativa de resultados:** métricas, validación cruzada y matrices de confusión.
6. **Interpretación de los modelos:** qué variables tienen más peso y cómo interpretar su influencia.
7. **Prueba con un cliente hipotético:** introducción de valores para comparar las predicciones de los tres modelos.
8. **Conclusiones:** qué diferencias he encontrado y qué tendría en cuenta al elegir un modelo.

## Código y metodología

He desarrollado el análisis con Python y scikit-learn, utilizando `LogisticRegression`, `RandomForestClassifier` y `GradientBoostingClassifier`.

Los resultados y las métricas se calculan mediante `model/train.py` y se guardan para que la aplicación pueda mostrarlos sin volver a entrenar los modelos cada vez que se abre.

En la prueba con un cliente hipotético utilizo los tres modelos entrenados para obtener sus predicciones y probabilidades, lo que permite comparar sus respuestas con los mismos datos de entrada.

**[Consultar el código y los notebooks en GitHub](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/clasificacion/04-comparativa-modelos/04-segmentacion-aerolinea)**

<details>
<summary>Estructura del proyecto</summary>

```text
app.py                    Aplicación y contenido de las secciones
components/
  editorial.py            Componentes para presentar el análisis
  charts.py               Gráficos y colores del proyecto
utils/
  data_loader.py           Carga de datos y modelos entrenados
  classifier.py             Predicciones para clientes hipotéticos
model/
  train.py                  Entrenamiento y evaluación de los modelos
  artifacts/                Métricas, matrices y modelos entrenados
data/                       Dataset original
assets/
  editorial.css             Estilos de composición
  style.css                 Colores y tipografías del proyecto

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
