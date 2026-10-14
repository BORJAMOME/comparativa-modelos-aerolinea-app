# Comparativa de modelos de Machine Learning: Segmentación de clientes de aerolínea

En este proyecto he comparado tres modelos de Machine Learning para clasificar a los clientes de una aerolínea en tres segmentos: Básico, Frecuente y Premium.

Mi objetivo ha sido comprobar qué diferencias hay entre la Regresión Logística, Random Forest y Gradient Boosting al trabajar con los mismos datos y criterios de evaluación. Para ello, he desarrollado una aplicación interactiva en la que se puede probar un cliente hipotético y comparar las predicciones de los tres modelos.

**[Abrir la aplicación interactiva](https://comparativa-modelos-aerolinea.streamlit.app)**

## El análisis

He trabajado con un conjunto de datos de 1.500 clientes y 21 variables originales. Antes de entrenar los modelos, he revisado la calidad de los datos, los valores atípicos, la multicolinealidad y los posibles problemas de fuga de información.

Después, he preparado y evaluado los tres algoritmos utilizando el mismo preprocesamiento, la misma partición de los datos y validación cruzada. De esta forma, he podido comparar sus resultados en condiciones equivalentes.

### Resultados

Gradient Boosting ha obtenido el mejor resultado de los tres modelos, con un F1-macro de 0,784 en validación cruzada y de 0,799 en el conjunto de test.

En validación cruzada, su resultado ha sido aproximadamente un 18 % superior al de la Regresión Logística. También he incluido un modelo de referencia (*baseline*) para comprobar la mejora obtenida frente a una predicción sencilla.

Más allá de las métricas, esta comparación me ha permitido ver que el rendimiento no es el único criterio que conviene tener en cuenta. La Regresión Logística, por ejemplo, facilita la interpretación de cómo se relacionan las variables con las predicciones. Esto puede ser útil cuando, además de clasificar a los clientes, necesitamos entender los motivos de cada clasificación.

## Qué encontrarás en la aplicación

He dividido la aplicación en ocho apartados para mostrar el análisis desde el planteamiento inicial hasta las conclusiones:

1. **El problema:** el objetivo de la segmentación y su posible utilidad para una aerolínea.
2. **Los datos:** las variables disponibles y los principales aspectos que he revisado.
3. **Preparación de los datos:** análisis de valores atípicos, multicolinealidad y posibles fugas de información.
4. **Preparación de los modelos:** el proceso que he seguido para compararlos en condiciones equivalentes.
5. **Comparativa de resultados:** métricas, validación cruzada y matrices de confusión.
6. **Interpretación de los modelos:** las variables más relevantes y su relación con las predicciones.
7. **Prueba con un cliente hipotético:** una forma de comparar las predicciones de los tres modelos a partir de los mismos datos.
8. **Conclusiones:** las principales diferencias que he encontrado y los factores que tendría en cuenta al elegir un modelo.

## Tecnologías y desarrollo

He desarrollado el análisis con Python y scikit-learn, utilizando `LogisticRegression`, `RandomForestClassifier` y `GradientBoostingClassifier`.

He separado el entrenamiento y la evaluación de los modelos de la aplicación interactiva. El archivo `model/train.py` calcula las métricas y genera los resultados que necesita la aplicación, de modo que no es necesario volver a entrenar los modelos cada vez que se abre.

En la prueba con un cliente hipotético, utilizo los tres modelos entrenados para obtener sus predicciones y probabilidades. Así, puedo observar cómo responde cada algoritmo ante los mismos datos de entrada.

**Borja Mora Méndez**

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
