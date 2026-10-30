# ¿Esta cara es real?
## Detección de rostros generados o manipulados por IA (deepfake) en imágenes mediante algoritmos clásicos de aprendizaje automático

![Banner del proyecto](banner-proyecto.jpg)

**Autores:** Juan Daniel Torres Ramírez, Sebastián Castellanos Prada

**Objetivo:** Determinar si la fotografía de un rostro es auténtica o fue manipulada por una inteligencia artificial, usando los algoritmos de aprendizaje automático y las redes neuronales vistos en clase.

Proyecto final de **Inteligencia Artificial 1, Grupo F1** · Universidad Industrial de Santander · 2026-2

---

## Dataset

**HiDF: A Human-Indistinguishable Deepfake Dataset** (Kang et al., KDD 2025)

**Descarga:** https://zenodo.org/records/16140829

| | |
|---|---|
| Imágenes reales | 38.582 archivos · 38.540 identidades (CelebA-HQ y FFHQ) |
| Imágenes falsas | 31.249 archivos (*face swap* con herramientas comerciales) |
| Resolución original | 1024 × 1024, rostros ya alineados y recortados |
| Metadatos | `metadata.csv` con raza, género y edad de cada rostro real |
| Licencia | CC BY-NC 4.0 |
| Usadas en el proyecto | 69.786 imágenes estandarizadas a 64 × 64 (Avance 1); submuestra de 12.000 para entrenar los clasificadores (Avance 2) |

Las imágenes falsas fueron curadas a mano por los autores del dataset para conservar únicamente
aquellas que un evaluador humano no distingue de una fotografía real.

---

## Modelos

**Procesamiento y análisis (Avance 1):** estandarización de imágenes a 64 × 64 ·
reducción por interpolación bilineal · imagen promedio · desviación estándar por píxel ·
escala de grises · normalización *z-score* · error estándar de la media ·
detección de fuga de etiqueta.

*Interpolación bilineal:* es el método con el que se redujo cada imagen de 1024 × 1024 a 64 × 64.
El valor de cada píxel nuevo se calcula como el promedio ponderado de los píxeles vecinos de la
imagen original, en lugar de copiar uno solo y descartar el resto, de modo que la imagen reducida
conserva el tono y los bordes del rostro.

*Normalización z-score:* es el paso que se aplica usando la imagen promedio y la de desviación ya
calculadas. A cada píxel se le resta el promedio y se le divide por la desviación,
`z = (x - promedio) / desviación`, y así todos los píxeles quedan con promedio 0 y desviación 1.
Sirve para que ninguna zona de la imagen pese más que otra solo porque sus valores son
numéricamente más grandes.

**Clasificación (Avance 2):** Naive Bayes Gaussiano (GNB) · árbol de decisión (DT) con curva de
`max_depth` · Random Forest (RF) con curva de `n_estimators` · máquina de soporte vectorial (SVM) con
kernels lineal, polinómico y RBF · tres perceptrones multicapa (DL-3, DL-6 y DL-10) con capas
ocultas `relu`, salida de 2 neuronas `softmax` y optimizador Adam, con un experimento por cada
número de epochs (5, 10, 50 y 100).

*Cómo se evita que el resultado salga inflado:* ningún modelo usa el peso ni el formato del archivo, y
todas las imágenes se vuelven a guardar en JPEG con la misma calidad (90); la partición 80/20 se hace
por foto base, de modo que una foto real y las falsas construidas sobre ella nunca quedan repartidas
entre entrenamiento y prueba; los hiperparámetros
se eligen con un conjunto de validación sacado del 80 % de entrenamiento, y la prueba se usa una sola
vez por modelo.

**Siguientes entregas:** aprendizaje no supervisado · reducción de dimensionalidad · clustering,
sobre las mismas imágenes.

---

## Herramientas y librerías

| Librería | Uso en el proyecto |
|---|---|
| **NumPy** | Manejo de los arreglos de imágenes y todos los cálculos: promedio, desviación estándar, percentiles y normalización |
| **pandas** | Lectura de `metadata.csv`, construcción de la tabla de medidas y estadística descriptiva |
| **Matplotlib** | Gráficas de torta, histogramas, imágenes promedio y de desviación, curvas de hiperparámetros y comparación de modelos |
| **scikit-learn** | Partición de los datos, GNB, árbol de decisión, Random Forest, SVM y cálculo del accuracy |
| **TensorFlow / Keras** | Construcción y entrenamiento de los perceptrones multicapa |
| **Pillow (PIL)** | Apertura, decodificación y redimensionado de las imágenes, y recompresión JPEG (Avance 2) |
| **zipfile, os, io, glob, time, shutil** | Librerías estándar de Python: lectura de los archivos comprimidos, recorrido de rutas y control del proceso por tandas |
| **IPython.display** | Visualización del banner dentro del notebook |
| **google.colab** | Montaje de Google Drive para guardar el resultado procesado |
| **aria2** | Descarga del dataset con múltiples conexiones en paralelo (herramienta de línea de comandos) |

Entorno de ejecución: **Python 3 sobre Google Colab**. El Avance 1 se ejecutó en la versión gratuita; el
Avance 2, en Colab Pro con GPU NVIDIA L4.

---

## Enlaces

| Recurso | Enlace |
|---|---|
| Repositorio | https://github.com/Judator07/ia1-deteccion-deepfakes |
| Código (notebook) | [`ProyectoClase_IA1_JuanTorres_SebastianCastellanos.ipynb`](ProyectoClase_IA1_JuanTorres_SebastianCastellanos.ipynb) |
| Dataset | https://zenodo.org/records/16140829 |
| Video | *Pendiente, entrega final* |
| Diapositivas | *Pendiente, entrega final* |

---

## Estado del proyecto

| Entrega | Estado |
|---|---|
| **Avance 1: exploración de datos** | Completado |
| **Avance 2: modelos de clasificación** | Completado |
| Entrega final | Pendiente |

---

## Resultados del primer avance

**Los datos.** Se descargó y procesó el dataset completo: **69.786 imágenes** (31.246 falsas y
38.540 reales), reducidas de 1024 × 1024 a **64 × 64**, lo que disminuye 256 veces el tamaño de cada
imagen y deja el conjunto en 858 MB. El dataset presenta un fuerte desbalance demográfico (75 % de
rostros de personas blancas y 60 % de mujeres), que queda registrado para evaluar más adelante si el
detector se comporta distinto entre grupos.

**Ninguna medida global separa las clases.** Brillo, contraste y los promedios de color producen
distribuciones superpuestas entre imágenes reales y falsas. La diferencia entre la imagen promedio de
cada clase es de **1,02 sobre 255**, que equivale a **2,7 veces el ruido de muestreo esperado**
(0,38): es una diferencia real, pero mínima. Era el resultado previsible, ya que el dataset fue
construido para ser indistinguible a simple vista.

**La diferencia está localizada.** El mapa de valor absoluto de la resta entre ambas imágenes
promedio se concentra en la región del rostro (ojos, nariz y boca) y es prácticamente nulo en el
fondo y el cabello, dado que cada imagen falsa conserva la fotografía original y solo sustituye el
rostro.

**Dos fugas de etiqueta detectadas y documentadas.** Una regla basada únicamente en el peso del
archivo alcanza un **97,5 %** de acierto, frente al 55,2 % de responder siempre «real», porque las
dos carpetas del dataset se guardaron con formatos y calidades distintos. Además, el **98,8 %** de
las imágenes falsas comparte su fotografía base con una imagen real del conjunto. Ambas situaciones
producirían un desempeño artificialmente alto y se corrigen normalizando la compresión y agrupando
por foto base al partir los datos.

---

## Resultados del segundo avance

Todos los modelos se entrenaron con el mismo 80 % (9.578 imágenes) y se midieron en el mismo 20 % de
prueba (2.422 imágenes), que ninguno vio antes. La referencia es responder siempre «real», que acierta
el **54,7 %**.

| Modelo | Hiperparámetro elegido | Accuracy en prueba |
|---|---|---|
| DL-10 (10 capas de 128) | 100 epochs | **77,9 %** |
| DL-3 (64, 128, 128) | 100 epochs | 77,1 % |
| DL-6 (64, 64, 128, 128, 256, 256) | 100 epochs | 77,0 % |
| SVM lineal | kernel lineal | 73,5 % |
| SVM RBF | kernel RBF | 69,7 % |
| Random Forest | 400 árboles | 66,8 % |
| SVM polinómico | kernel polinómico (grado 3) | 60,0 % |
| Árbol de decisión | `max_depth` = 6 | 59,8 % |
| Naive Bayes Gaussiano | sin hiperparámetros | 52,6 % |

**Las redes neuronales son las mejores** y reducen el error a menos de la mitad (22,1 % contra 45,3 %
de la referencia). Entre ellas la diferencia es menor a un punto, unas 22 imágenes, así que no alcanza
para decir que más capas sea mejor. Las redes sobreajustan (95 a 96 % en entrenamiento), pero su
acierto en prueba es casi igual al de validación.

**Entre los modelos clásicos gana la SVM lineal.** Con la configuración usada, las fronteras más
flexibles (`rbf`, `poly`) quedan por debajo incluso en entrenamiento, y el árbol profundo llega a
99,6 % en entrenamiento pero cae a 56,1 % en validación. **GNB queda por debajo de la referencia**, porque supone que los píxeles son independientes y
en una imagen cada píxel se parece a sus vecinos.

---

## Estructura del repositorio

```
├── README.md
├── banner-proyecto.jpg                                          Banner del proyecto (800 × 300)
└── ProyectoClase_IA1_JuanTorres_SebastianCastellanos.ipynb  Notebook del proyecto (Avance 1 y Avance 2)
```

---

## Ejecución

El notebook está preparado para **Google Colab**. Requiere una cuenta de Google Drive con espacio
disponible para almacenar el resultado procesado (unos 900 MB).

1. Abrir `ProyectoClase_IA1_JuanTorres_SebastianCastellanos.ipynb` en Google Colab.
2. Ejecutar las celdas en orden. La sección 3 descarga los archivos comprimidos del dataset (40 GB)
   y los reduce a 64 × 64; **se ejecuta una sola vez** y toma aproximadamente 45 minutos.
3. Las secciones 4 a 11 trabajan sobre el archivo reducido guardado en Drive y se ejecutan en
   pocos minutos.
4. El Avance 2 empieza en la sección 13 y no necesita repetir la descarga: basta ejecutar la primera
   celda y luego de la sección 13 en adelante. Toma alrededor de hora y media, casi toda por la SVM;
   para las redes neuronales conviene activar una GPU (Entorno de ejecución, Cambiar tipo de entorno
   de ejecución).

Las librerías de Python vienen preinstaladas en Colab; la única dependencia adicional es `aria2`,
que el propio notebook instala.

---

## Referencias

- Kang, C., Jeong, S., Lee, J., Choi, D., Woo, S. S. y Han, J. (2025). *HiDF: A
  Human-Indistinguishable Deepfake Dataset*. Proceedings of the 31st ACM SIGKDD Conference on
  Knowledge Discovery and Data Mining (KDD '25), Datasets and Benchmarks Track.
  https://doi.org/10.1145/3711896.3737399

- Karras, T., Aila, T., Laine, S. y Lehtinen, J. (2018). *Progressive Growing of GANs for Improved
  Quality, Stability, and Variation*. International Conference on Learning Representations (ICLR).
  [Origen del conjunto CelebA-HQ]

- Karras, T., Laine, S. y Aila, T. (2019). *A Style-Based Generator Architecture for Generative
  Adversarial Networks*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
  [Origen del conjunto FFHQ]

- Garzón Villamizar, G. (2026). *Inteligencia Artificial: material del curso* [Repositorio de
  clase]. Universidad Industrial de Santander.
  https://gitlab.com/bivl2ab/academico/cursos-uis/ai/ai-uis-student

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
  Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M.,
  Perrot, M. y Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine
  Learning Research, 12*, 2825-2830.

- Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G.,
  Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P.,
  Vasudevan, V., Warden, P., Wicke, M., Yu, Y. y Zheng, X. (2016). TensorFlow: A system for
  large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and
  Implementation (OSDI 16)*, 265-283.

- Garzón Villamizar, G. (agosto a octubre de 2026). *Clases de Inteligencia Artificial 1*
  [Comunicación personal]. Universidad Industrial de Santander, Bucaramanga, Colombia.

---

*El dataset HiDF se distribuye bajo licencia CC BY-NC 4.0 y se emplea aquí con fines exclusivamente
académicos.*
