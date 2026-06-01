# Clasificador de basura con TensorFlow y Keras
Carlos Iván Fonseca Mondragón | A01771689

Este proyecto busca desarrollar un modelo de clasificación de imágenes de basura utilizando TensorFlow. El modelo toma imágenes de basura como entrada y las clasifica en diferentes categorías (plástico, papel, metal, etc.). El dataset fue tomado de Kaggle [WasteWise: Know Your Waste](https://www.kaggle.com/datasets/smarthkaushal/waste-segregation) y se preprocesa utilizando técnicas de aumento de datos para mejorar la generalización del modelo.

## Estructura del proyecto

- ETL_Preprocesamiento.ipynb: Notebook para el preprocesamiento de datos, incluyendo limpieza, normalización y aumento de datos.
- Modelo_Entrenamiento_v3.ipynb: Notebook para la construcción, entrenamiento y evaluación del modelo de clasificación.
- queries.py: Script para realizar predicciones utilizando el modelo entrenado, utilizando una interfaz gráfica con Tkinter.
- train/ : Carpeta que contiene el dataset de entrenamiento.
- test/ : Carpeta que contiene el dataset de prueba.
- test_cases/ : Carpeta con casos de prueba para validar el modelo y la interfaz de predicción.

## Requisitos

- Python 3.11 para asegurar paridad con la versión utilizada en el entorno de desarrollo.
- TensorFlow 2.x para la construcción y entrenamiento del modelo.
- Tkinter para la interfaz gráfica de predicción.
- Joblib para cargar los metadatos preprocesados.
- Matplotlib para visualización de resultados.
- Scikit-learn para métricas de evaluación.
- NumPy para manipulación de datos.
- Seaborn para visualización de datos.
- OS para manejo de archivos y directorios.
- PIL para procesamiento de imágenes.

## Reproducción del proyecto
1. Clonar el repositorio y navegar a la carpeta del proyecto.
2. Descargar el dataset desde Kaggle, y colocar las carpetas de "train" y "test" en el directorio raíz del proyecto, junto a los notebooks y el script de queries.
3. Instalar las dependencias necesarias utilizando pip.
4. Ejecutar el notebook `ETL_Preprocesamiento.ipynb`.
5. Ejecutar el notebook `Modelo_Entrenamiento_v3.ipynb` para entrenar el modelo y evaluar su desempeño.
6. Ejecutar el script `queries.py` para probar la interfaz de predicción con imágenes de prueba.

## Dataset

El dataset utilizado es "WasteWise", un conjunto de datos de imágenes de basura con 4765 imágenes distribuidas en 9 categorías: cartón, vidrio, metal, papel, plástico, residuos orgánicos, textiles, vegetación y otros. Este dataset ya está dividido en conjuntos de entrenamiento y prueba, lo que facilita el proceso de desarrollo del modelo. Sin embargo, cuenta con un número limitado y desequilibrado de imágenes, lo que presenta desafíos para el entrenamiento de modelos de aprendizaje profundo, por lo que se aplicaron técnicas de aumento de datos para mejorar la generalización del modelo. Según el autor del dataset, el propósito de este conjunto de datos es apoyar el desarrollo de soluciones de IA sostenibles y fomentar la conciencia sobre la gestión adecuada de residuos.

## Preprocesado del dataset

El preprocesamiento del dataset se realizó en el notebook `ETL_Preprocesamiento.ipynb`, donde se aplicaron las siguientes técnicas:

- Establecimiento de un tamaño estándar para las imágenes (224x224 píxeles) para asegurar la consistencia en el entrenamiento del modelo.
- Creación de directorios para almacenar los datasets preprocesados de entrenamiento, validación y prueba.
- Establecer una semilla para garantizar la reproducibilidad de los resultados, así como un tamaño de lote de 32 para el entrenamiento.
- Limpieza de datos: Detección y eliminación de imágenes corruptas.
- Creación de datasets iniciales utilizando `image_dataset_from_directory` de TensorFlow, con una división del 90% del set de entrenamiento para entrenar y el 10% restante para validación, asegurando que el modelo pueda generalizar mejor a datos no vistos durante el entrenamiento, de esta manera, se utilizan 3426 imágenes para entrenamiento y 380 para validación.
- Normalización: Escalado de los valores de píxeles a un rango de [0, 1], pues son imagenes RGB con valores en el rango [0, 255], esto se aplica tanto a las imágenes de entrenamiento como a las de prueba para asegurar que el modelo reciba datos consistentes durante el entrenamiento y la evaluación.
- Guardar los datasets preprocesados con la función save() de TensorFlow para su uso posterior en el entrenamiento del modelo.
- Generación de diccionario con pesos por clase para manejar el desequilibrio, aunque esta técnica fue deprecada en el modelo final debido a que generó un sobreajuste.
- Guardar los metadatos del preprocesamiento, incluyendo los nombres de las clases y el diccionario de pesos por clase, utilizando Joblib para su uso posterior en la fase de predicción, así como valores de tamaño de lote, semilla y número de clases para asegurar la consistencia entre el preprocesamiento y la predicción.

Adicionalmente, en esta notebook se muestran ejemplos de imagenes preprocesadas con normalización aplicada.

## Construcción, entrenamiento y evaluación del modelo

Tomando como base la arquitectura de 5 capas convolucionales propuesta por Nnamoko et al. (2022) en el artículo [Solid Waste Image Classification Using Deep Convolutional
Neural Network](https://www.mdpi.com/2412-3811/7/4/47), se adaptó el modelo para el problema de clasificación en 9 categorías. 

El modelo se construyó utilizando TensorFlow y Keras en el notebook `Modelo_Entrenamiento_v3.ipynb`. Antes de entrenar el modelo, se cargaron los datasets preprocesados y los metadatos generados en la fase de preprocesamiento. Se aplicó una estrategia de data augmentation a través de un pipeline con Keras que incluye rotaciones, zoom y contraste aleatorio para mejorar la generalización del modelo; evitando el uso de otras técnicas como el Gaussian Blur, que aunque puede ayudar a reducir el ruido, también puede eliminar detalles importantes de las imágenes, lo que podría afectar negativamente el rendimiento del modelo. Este proceso se aplicó solo al dataset de entrenamiento para evitar introducir ruido en los datos de validación y prueba.

La red recibe imágenes de entrada con dimensiones de 224×224×3 píxeles, normalizadas al rango [0, 1] mediante una capa de reescalado aplicada durante el preprocesamiento. La arquitectura cuenta con cinco capas secuenciales, cada una compuesta por una capa Conv2D con kernel de 3×3 y padding same, seguida de BatchNormalization y una capa de MaxPooling2D de 2×2. Los filtros de cada bloque aumentan progresivamente, con 32, 64, 128, 256 y 512 filtros respectivamente, duplicando la capacidad para representar características en cada nivel, permitiendo que las capas más profundas capturen patrones de mayor abstracción.
La red recibe imágenes de entrada con dimensiones de 224×224×3 píxeles, normalizadas al rango [0, 1] mediante una capa de reescalado aplicada durante el preprocesamiento. La arquitectura cuenta con cinco capas secuenciales, cada una compuesta por una capa Conv2D con kernel de 3×3 y padding same, seguida de BatchNormalization y una capa de MaxPooling2D de 2×2. Los filtros de cada bloque aumentan progresivamente, con 32, 64, 128, 256 y 512 filtros respectivamente, duplicando la capacidad para representar características en cada nivel, esto permite que las capas más profundas capturen patrones de mayor abstracción.

A diferencia de la arquitectura original, y de otras arquitecturas comunes como AlexNet, que emplean Flatten para convertir las características extraídas en un vector de una dimensión antes de la capa densa, se optó por utilizar GlobalAveragePooling2D. Esto reduce drásticamente el número de parámetros, lo que resulta en un menor riesgo de sobreajuste, especialmente dado el tamaño limitado del dataset. La salida de esta capa se conecta a una capa densa con 256 unidades y activación ReLU, que también emplea regularización L2, seguida de una capa de Dropout con una tasa del 60% para mitigar aún más el riesgo de sobreajuste.

La capa de salida cuenta con 9 neuronas, correspondientes a las 9 categorías de basura, y utiliza la función de activación softmax para generar probabilidades de clasificación.

El modelo se compiló utilizando el optimizador Adam, conocido por su adaptabilidad y eficiencia en la convergencia, con una tasa de aprendizaje inicial de 5e-5. Como función de pérdida se utilizó CategoricalCrossentropy, adecuada para problemas de clasificación multiclase.

Durante el entrenamiento se emplearon tres callbacks: ModelCheckpoint para preservar los pesos del modelo con mejor val_accuracy, ReduceLROnPlateau para reducir la tasa de aprendizaje a la mitad cuando val_accuracy no mejora en 15 épocas consecutivas, y EarlyStopping con paciencia de 25 épocas para detener el entrenamiento ante ausencia de mejora sostenida.

El modelo se entrenó durante 50 épocas con un tamaño de lote de 32, el mismo definido en los metadatos del preprocesamiento.

## Métricas de evaluación

Para evaluar el desempeño del modelo propuesto, se emplearon las métricas estándar utilizadas en problemas de clasificación supervisada: exactitud global (accuracy), precisión (precision), exhaustividad (recall) y puntuación F1 (F1-score). Estas métricas fueron calculadas tanto a nivel global como por clase individual, puesto que el dataset presenta un desbalance moderado entre categorías, con clases que van desde aproximadamente 230 hasta 730 imágenes en el conjunto de entrenamiento, lo cual puede afectar el rendimiento real del modelo si se evalúa únicamente con accuracy global.

La exactitud global se define como la proporción de predicciones correctas sobre el total de muestras evaluadas:

AC = T_r/T_n

donde T_r representa el número de muestras clasificadas correctamente y T_n el total de muestras en el conjunto de prueba. Si bien esta métrica ofrece una visión general del rendimiento, resulta insuficiente en contextos de desbalance de clases, ya que un modelo que prediga sistemáticamente las clases mayoritarias puede alcanzar valores de accuracy engañosamente altos.

Por esta razón, se complementó la evaluación con precisión, recall y F1-score por clase. La precisión por clase indica qué proporción de las predicciones asignadas a esa clase fueron correctas.

El recall por clase mide qué proporción de las muestras reales de esa clase fueron identificadas correctamente por el modelo.

El F1-score por clase combina precisión y recall en una única métrica mediante su media armónica, resultando especialmente útil para evaluar el rendimiento en clases desbalanceadas, ya que penaliza tanto los falsos positivos como los falsos negativos (Sokolova & Lapalme, 2009).

Más adelante, se imprime un reporte de clasificación detallado utilizando la función `classification_report` de Scikit-learn, que muestra estas métricas para cada clase, así como un promedio ponderado que refleja el rendimiento general del modelo teniendo en cuenta el desbalance entre clases.

Finalmente, se generó una matriz de confusión sobre el conjunto de prueba para analizar los patrones en lo que respecta a las categorías que tienden a confundirse entre sí. Esto permite identificar el tipo de error que presenta el modelo.

## Resultados

El modelo fue evaluado utilizando el las 957 imágenes del conjunto de prueba, logrando una exactitud global de 55.49%, con una pérdida de 1.6172. Durante el entrenamiento, se alcanzó una mejor val_accuracy de 58.95%. Estos resultados son relativamente consistentes con lo reportado por Nnamoko et al. (2022) para CNNs entrenadas desde cero, considerando que el presente trabajo aborda un problema más complejo al clasificar nueve categorías en lugar de dos.

Las curvas de entrenamiento muestran una convergencia de manera progresiva con un nivel moderado de overfitting, alcanzando una diferencia final de cerca de 10 puntos entre train y val accuracy, junto con oscilaciones en val_loss, que pueden atribuirse al desbalance de clases del dataset.

<img width="1389" height="490" alt="graph1" src="https://github.com/user-attachments/assets/a6a8ef76-ce86-42ae-b628-182f894945c8" />

(Figura 1. Gráficas de Accuracy y Loss por Epoch)

Al analizar cada clase, se observa un comportamiento heterogéneo. Las clases Textile Trash, Food Organics, Glass y Paper obtuvieron las precisiones más altas, con valores entre 0.73 y 0.79, pero con recall bajo (0.23–0.31), lo que indica que el modelo predice estas clases con certeza pero omite muchas de sus instancias reales. En comparación, las clases Vegetation y Plastic tienen un patrón opuesto, con recall alto (0.99 y 0.77) y una precisión moderada (0.46 y 0.49), consistente con su mayor representación en el dataset. La categoría con peor desempeño fue Miscellaneous Trash, con un F1-score de 0.24, lo cual es esperado puesto que agrupa basura sin ninguna característica visual distintiva, lo que dificulta su clasificación, como se puede ver en la matríz de confusión: 

<img width="934" height="790" alt="cf" src="https://github.com/user-attachments/assets/69ca78fd-29d1-4c0f-b76d-fb5e413d4c7f" />
(Figura 2. Matríz de confusión True vs Predicted)



El F1-score macro promedio fue 0.52, reflejando la disparidad entre clases.

## Queries
Se adjunta un archivo **queries.py**, que te permite realizar predicciones con el modelo, basta con descargar el archivo .py y el modelo, asegurarte de que estén en el mismo directorio y podrás ver la predicción arrojada por el modelo. También se adjunta una carpeta con imágenes para probar, nunca antes vistas por el modelo. Es requisito contar con las mismas librerías que se quieren en los archivos de ETL y generación del modelo.

<img width="798" height="910" alt="image" src="https://github.com/user-attachments/assets/3cf08dcc-9d67-4378-9806-3d9996bea7e7" />
(Figura 3. Uso del archivo queries.py)

## Conclusiones

Este proyecto logró desarrollar un modelo de clasificación de imágenes de basura utilizando TensorFlow, basado en la arquitectura de cinco bloques convolucionales de Nnamoko et al. (2022) y adaptado con Global Average Pooling, regularización L2 y Dropout, alcanzando una exactitud global del 55.49% en el conjunto de prueba. Aunque este resultado no es excepcional, es consistente con lo reportado en la literatura para modelos entrenados desde cero en problemas de clasificación multiclase con datasets limitados y desbalanceados.

Como mejora, se puede explorar el uso de transfer learning con arquitecturas preentrenadas en ImageNet, lo que podría mejorar significativamente el rendimiento al aprovechar características previamente aprendidas. Además, se podrían aplicar técnicas de manejo de desbalance de clases más avanzadas, o incluso agregar más datos al dataset para mejorar la representación de las clases minoritarias. En general, este proyecto demuestra la viabilidad de utilizar CNNs para la clasificación de basura, aunque también destaca los desafíos asociados con datasets limitados y desbalanceados.

## Referencias

1. Lv, Q., Zhang, S., & Wang, Y. (2022). Deep learning model of image classification using machine learning. Advances in Multimedia, 2022, Artículo 3351256. https://doi.org/10.1155/2022/3351256

2. Nnamoko, N., Barrowclough, J., & Procter, J. (2022). Solid waste image classification using deep convolutional neural network. Infrastructures, 7(4), Artículo 47. https://doi.org/10.3390/infrastructures7040047

3. Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A novel real-life data set for landfill waste classification using deep learning. Information, 14(12), Artículo 633. https://doi.org/10.3390/info14120633

4. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
