# Clasificador de basura con TensorFlow y Keras
Carlos Iván Fonseca Mondragón | A01771689

Este proyecto busca desarrollar un modelo de clasificación de imágenes de basura utilizando TensorFlow. El modelo toma imágenes de basura como entrada y las clasifica en diferentes categorías (plástico, papel, metal, etc.). El dataset fue tomado de Kaggle [WasteWise: Know Your Waste](https://www.kaggle.com/datasets/smarthkaushal/waste-segregation) y se preprocesa utilizando técnicas de aumento de datos para mejorar la generalización del modelo.

## Estructura del proyecto

- ETL_Preprocesamiento.ipynb: Notebook para el preprocesamiento de datos, incluyendo limpieza, normalización y aumento de datos.
- Modelo_Entrenamiento_v3.ipynb: Notebook para la construcción, entrenamiento y evaluación del modelo de clasificación inicial
- Modelo_Entrenamiento_v4: Notebook para el modelo con un ajuste en filtros y en tasa de aprendizaje.
- Modelo_Entrenamiento_v5_MobileNet: Notebook que emplea un modelo base de MobileNetV2 y transfer learning para mejorar las clasificaciones.
- queries.py: Script para realizar predicciones utilizando el modelo entrenado, utilizando una interfaz gráfica con Tkinter.
- check_test_cases.py: Script para realizar múltiples inferencias a la vez, no cuenta con interfaz gráfica.
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
5. Ejecutar el notebook correspondiente al modelo que se deseé evaluar.
6. Ejecutar el script `queries.py` o `check_test_cases.py`  para probar la interfaz de predicción con imágenes de prueba.

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

## Construcción, entrenamiento y evaluación del modelo inicial

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

Este modelo se guardó como `best_model.keras`, y se encuentra en la notebook "Modelo_Entrenamiento_v3.ipynb".

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

Para obtener una métrica de evaluación adicional, se creó un conjunto de imágenes de prueba nunca antes vistas por el modelo, tomadas de internet, con el objetivo de evaluar su capacidad de generalización a datos completamente nuevos. Estas imágenes se clasificaron utilizando el modelo entrenado, y se observó que, aunque el modelo logró clasificar correctamente algunas imágenes (4/18), cometió errores significativos, asignando la mayoría de predicciones a las clases de Vegetation, Cardboard y Paper, lo que sugiere que el modelo tiene dificultades para generalizar a imágenes con características visuales diferentes a las del dataset de entrenamiento. Analizando las imagenes del dataset nuevo, se observa que muchas de ellas presentan características visuales distintas a las del dataset original, como diferentes ángulos de captura, iluminación, fondos variados y otros objetos presentes en la escena, lo que puede haber contribuido a la dificultad del modelo para clasificarlas correctamente. Este fenómeno es conocido como _"domain shift"_ y es un desafío común en problemas de clasificación de imágenes, especialmente cuando el modelo fue entrenado con un dataset limitado y no representativo de la diversidad visual presente en el mundo real.

<img width="1536" height="696" alt="image" src="https://github.com/user-attachments/assets/3ffc7f88-b85c-47ce-8402-009bb6c5ba46" />

(Figura 3. Evaluación del modelo base)


## Mejoras implementadas al modelo

### Modelo con sexta capa convolucional y ajuste de learning rate
Se tomaron dos rutas de mejora para el modelo inicial, la primera mejora fue tomada a partir de una sugerencia del propio Nammoko et al. (2022), donde argumenta que el número de capas convolucionales afecta la calidad de las características extraídas, por lo que se agregó una sexta capa convolucional con 1024 filtros, seguida de BatchNormalization y MaxPooling2D, manteniendo la misma estructura de kernel y padding. La adición de esta capa implicó también un ajuste en la tasa de aprendizaje, pasando de 5e-5 a 1e-4, lo que permitió una convergencia más rápida durante el entrenamiento. Este modelo se guardó como `best_model_experimental.keras`, y se encuentra en la notebook "Modelo_Entrenamiento_v4.ipynb".

### Modelo con MobileNetV2 como base
La segunda mejora se basó en la implementación de transfer learning utilizando MobileNetV2 como base, siguiendo la metodología propuesta por Yong et al. (2023), donde demostraron que un modelo de clasificación de residuos basado en MobileNetV2 supera en 15.42 puntos porcentuales a una CNN entrenada desde cero, alcanzando una precisión absoluta del 82.92%. En este caso, se utilizó MobileNetV2 preentrenada como extractor de características, congelando sus pesos durante una primera fase de entrenamiento, y agregando una cabeza de clasificación personalizada con capas GlobalAveragePooling2D, una capa densa con 256 unidades con activación
ReLU y regularización L2 1e-4, seguida de una capa de Dropout con tasa del 0.5 y una capa de salida con 9 neuronas y activación softmax. 

A diferencia de la mejora anterior, que contaba con seis capas convolucionales, con una progresión de 32 a 1024 filtros, Dropout de 0.5 (a diferencia del 0.6 utilizado en los modelos anteriores) y sin un modelo preentrenado, esta segunda mejora deja la extracción de características a MobileNetV2, que ha sido entrenada en un dataset de millones de imágenes, le permite aprender patrones visuales complejos y generalizables, lo que reduce el riesgo de domain shift. 

El entrenamiento del modelo se separó en dos fases: una primera fase de entrenamiento con el modelo base congelado durante 15 épocas con una tasa de aprendizaje de 1e-3, permitiendo que la cabeza de clasificación se adapte a las características extraídas por MobileNetV2. En la segunda fase, enfocada al fine-tuning, se descongelaron las últimas 30 capas del modelo base y se continuó el entrenamiento por 55 épocas adicionales, con una tasa de aprendizaje reducida a 1e-5, esto para evitar que los pesos preentrenados se modifiquen de manera drástica. Ambas fases de entrenamiento utilizaron los mismos callbacks que el modelo inicial, con ModelCheckpoint para guardar el mejor modelo basado en val_accuracy, ReduceLROnPlateau para reducir la tasa de aprendizaje cuando no se observe mejora en val_accuracy a lo largo de 15 épocas consecutivas, y EarlyStopping con paciencia de 25 épocas, monitoreando val_accuracy.

Este modelo se guardó como `best_model_mobilenet.keras`, y se encuentra en la notebook "Modelo_Entrenamiento_v5.ipynb".

## Evaluación de las mejoras

### Modelo con sexta capa convolucional y ajuste de learning rate
El modelo con la sexta capa convolucional logró una mejora marginal en la exactitud global, alcanzando un 63.32% en el conjunto de prueba, lo que representa una mejora de aproximadamente 7.83 puntos porcentuales respecto al modelo inicial. Su exactitud en el conjunto de validación durante el entrenamiento alcanzó un máximo de 63.95%, lo que indica una mejor capacidad de generalización. 

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/711e03c4-cfb2-4850-97e6-9a475817e081" />

(Figura 4. Gráficas de Accuracy y Loss por Epoch en el modelo v4)

Al analizar cada clase del modelo, se observa una mejora generalizada respecto al modelo inicial, con un compartimiento más balaanceado entre precisión y recall. Las clases Textile Trash y Glass lograron las precisiones más altas, con valores de 0.91 y 0.94, respectivamente, aunque con recall bajo (0.33 y 0.58), esto indica que el modelo predice estas clases con alta certeza, pero sigue sin identificar muchas de sus instancias reales. Este comportamiento es similar para Food Organics, con precisión de 0.70 y recall de 0.47.

Vegetation, por otra parte, tiene un recall muy elevado (0.99), con una precisión moderada (0.64). Paper y Cardboard muestran el comportamiento más balanceado del modelo, con precisión y recall similares entre sí (0.63/0.79 y 0.63/0.80 respectivamente), lo que se refleja en sus F1-scores relativamente altos (0.70 y 0.71). El mayor avance de este modelo respecto al modelo inicial se observa en la clase Miscellaneous Trash, con un F1-score que pasa de 0.24 a 0.56, esto sugiere que la adición de la sexta capa de la red permitió capturar patron más abstractos propios de esta clase, aunque sigue siendo la clase con peor desempeño.

<img width="934" height="790" alt="image" src="https://github.com/user-attachments/assets/65df5f7b-7073-46fe-bdea-b56c0bedc14b" />

(Figura 5. Matríz de confusión del modelo v4)

Evaluar este modelo con el conjunto de imágenes de prueba generado a partir de internet cuenta otra historia, pues logró clasificar correctamente 4 de las 18 imágenes, el mismo resultado que el modelo inicial, aunque con una distribución de predicciones diferente, pues ahora 10 de las 18 imágenes fueron clasificadas como Paper, y otras 5 como Cardboard, lo que sugiere que el modelo tiene una tendencia a clasificar imágenes nuevas en estas categorías, lo que indica que el modelo es suceptible, en la misma medida que el modelo inicial, al fenómeno de domain shift, lo que limita su capacidad de generalización a datos completamente nuevos.

<img width="1518" height="688" alt="image" src="https://github.com/user-attachments/assets/e91b3cd5-4823-47b2-8f07-fa606af857c6" />

(Figura 6. Evaluación del modelo v4)

### Modelo con MobileNetV2 como base
El modelo basado en MobileNetV2 representa la mejora más significativa del proyecto, alcanzando una exactitud global del 81.09% en el conjunto de prueba, un incremento de 25.6 puntos porcentuales respecto al modelo inicial, y una mejora de 12.96 puntos porcentuales contra el modelo con la sexta capa convolucional. Su exactitud máxima en el conjunto de validación durante el entrenamiento fue 85%, lo que indica una excelente capacidad de generalización, especialmente considerando el tamaño limitado del dataset. La caída en las curvas alrededor de la época 15 corresponden a la transición entre la fase de entrenamiento con el modelo base congelado y la fase de fine-tuning, donde se descongelaron las últimas 30 capas de MobileNetV2, lo que genera una perturbación inicial en el proceso de entrenamiento, pero que posteriormente permite una mejora significativa en el rendimiento del modelo.

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/cf2ea8fc-c480-4a63-9a16-28eac05e0390" />

(Figura 7. Gráficas de Accuracy y Loss por Epoch en el modelo con MobileNetV2)

El análisis por clase muestra una mejora sustancial en todas las categorías. Todas las categorías lograron un F1-score superior a 0.70, incluyendo a Miscellaneous Trash, que aunque sigue siendo la clase con peor desempeño, logró un F1-score de 0.71, una mejora significativa respecto a los modelos anteriores. Textile Trash, la clase que mostró más conflictos en los modelos anteriores, logró un F1-score de 0.84, contrasta con el 0.26 del modelo con la sexta capa convolucional, lo que sugiere que gracias a MobileNetV2, el modelo identifica más instancias reales de esta clase. Metal y Vegetation obtienen los F1-scores más altos (0.85 y 0.86 respectivamente), con Vegetation manteniendo su recall alto (0.98). Food Organics es la única clase que presenta una disparidad en precision/recall notable (0.89/0.69), con 17 de sus 83 instancias confundidas con Vegetation según la matriz de confusión. 

<img width="934" height="790" alt="image" src="https://github.com/user-attachments/assets/e7abe3fe-219f-4e03-b45d-5243d3597765" />

(Figura 8. Matriz de confusión del modelo MobileNetV2)

Al evaluar este modelo con el conjunto de imágenes de prueba generado a partir de internet, se obtienen 7 de 18 aciertos; una mejora significativa respecto a los modelos anteriores, con una distribución de predicciones diversa. El modelo clasificó correctamente imágenes de las clases Cardboard, Food Organics, Paper, Plastic y Vegetation, pero sigue fallando con Glass y Metal, que se clasifican erróneamente como Vegatation con confianzas de 99.07 y 100%, respectivamente. Textile Trash se dispersa entre Miscellaneous y Paper. Este resultado sugiere que el modelo basado en MobileNetV2 tiene una mejor capacidad de generalización a datos completamente nuevos, aunque sigue siendo susceptible al fenómeno de domain shift; las características visuales de materiales cuya apariencia puede variar significativamente dependiendo de factores como el ángulo de captura, la iluminación y el contexto, como Glass y Metal, siguen representando un desafío para el modelo.

<img width="1526" height="698" alt="image" src="https://github.com/user-attachments/assets/095e5087-bfa7-4df0-b02d-31d431a44b6a" />


(Figura 9. Evaluación del modelo con MobileNetV2)



## Queries
Se adjunta un archivo **queries.py**, que te permite realizar predicciones con el modelo, basta con descargar el archivo .py y el modelo, asegurarte de que estén en el mismo directorio y podrás ver la predicción arrojada por el modelo. También se adjunta una carpeta con imágenes para probar, nunca antes vistas por el modelo. Es requisito contar con las mismas librerías que se quieren en los archivos de ETL y generación del modelo.

<img width="798" height="910" alt="image" src="https://github.com/user-attachments/assets/3cf08dcc-9d67-4378-9806-3d9996bea7e7" />

(Figura 10. Uso del archivo queries.py)

Si deseas probar un lote de imágenes, puedes colocar las imágenes dentro de la carpeta `test_cases/` y ejecutar el script `check_test_cases.py`, el cual cargará el modelo, procesará cada imagen dentro de la carpeta `test_cases/` y mostrará la predicción para cada una de ellas. Revisa que las imágenes tengan un formato compatible y que estén nombradas ya sea con el formato `class_name_#.jpg`, `class_name_#.png`, o que el nombre cuente con al menos los primeros 4 caracteres del nombre de la clase a la que pertenecen, esto para facilitar la interpretación de los resultados.

## Conclusiones

Este proyecto logró desarrollar y comparar tres modelos de clasificación de imágenes de basura en nueve categorías, partiendo de una arquitectura CNN entrenada desde cero basada en Nnamoko et al. (2022) y progresando hasta una solución basada en transfer learning con MobileNetV2.

El modelo inicial alcanzó una exactitud global de 55.49%, consistente con artículos sobre CNNs entrenadas desde cero con datasets limitados. La primera mejora, añadiendo un sexto bloque convolucional y ajustando la tasa de aprendizaje, incrementó la exactitud a 63.32%, con avances notables en categorías como Miscellaneous Trash, con un F1 que pasó de 0.24 a 0.56. Sin embargo, ambos modelos mostraron áreas de mejora frente al domain shift, acertando apenas 4 de 18 imágenes externas, evidenciando que la baja variablidad del fondo en el dataset llevó al modelo a aprender correlaciones erróneas entre el contexto visual y la categoría del objeto.

El modelo con MobileNetV2 representó el avance más significativo, con una exactitud global de 81.09%, es decir, 25.6 puntos porcentuales sobre el modelo inicial, con todas las clases superando un F1-score de 0.70. La evaluación con imágenes externas mejoró a 7 de 18 aciertos, aunque Glass y Metal continuaron fallando debido a la alta variabilidad de su apariencia en condiciones no controladas.

Como trabajo futuro se sugiere explorar técnicas de domain adaptation, augmentation de fondo y datasets más diversos para reducir la susceptibilidad al domain shift. Este proyecto confirma que el transfer learning es una alternativa significativamente más robusta que el entrenamiento desde cero para problemas de clasificación de residuos con datasets de tamaño moderado.

## Referencias

1. Lv, Q., Zhang, S., & Wang, Y. (2022). Deep learning model of image classification using machine learning. Advances in Multimedia, 2022, Artículo 3351256. https://doi.org/10.1155/2022/3351256

2. Nnamoko, N., Barrowclough, J., & Procter, J. (2022). Solid waste image classification using deep convolutional neural network. Infrastructures, 7(4), Artículo 47. https://doi.org/10.3390/infrastructures7040047

3. Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A novel real-life data set for landfill waste classification using deep learning. Information, 14(12), Artículo 633. https://doi.org/10.3390/info14120633

4. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

5. Yong, L., Ma, L., Sun, D., & Du, L. (2023). Application of MobileNetV2 to waste classification. PLOS ONE, 18(3), Artículo e0282336. https://doi.org/10.1371/journal.pone.0282336
