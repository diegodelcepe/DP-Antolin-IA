# Documentación Completa del Backend (Aplicación Web + API de Inferencia de Anomalías)

## 1. Introducción General
  Este proyecto implementa un sistema completo de `detección de anomalías` en piezas industriales mediante vision artificial.
Se compone de un *`backend de inferencia`* basado en PathCore y un *`dataset`* especializado que entrena y alimenta dicho modelo.

## 1.1 Backend - API de Inferencia de Anomalías
  Este backend implementa una **API de inferencia de anomalías** basada en **PatchCore** sobre imágenes, construida con **FastAPI**.  
Expone endpoints para verificar el estado del sistema, servir un frontend estático y realizar predicciones de anomalía.  

Internamente carga un **backbone ResNet18**, un **memory bank** (KNN sobre embeddings normalizados) y aplica opcionalmente una **Región de Interés (ROI)** junto con operaciones de visualización (overlays y polígonos).

>**Nota:** Una API de inferencia es una interfaz que permite enviar datos (en este caso, imágenes) a un modelo de inteligencia artificial ya entrenado y recibir resultados o predicciones.

### Objetivos principales
- Recibir una imagen.  
- Transformarla y extraer características.  
- Calcular un mapa de calor de anomalías y un score.  
- Decidir si es anómala en función de umbrales flexibles.  
- *(Opcional)* Generar visualizaciones y polígonos de áreas anómalas para el frontend.


## 1.2 Dataset
  El dataset utilizado se denomina `dataset_gua_crops`, contiene imágenes reales de piezas plásticas realizadas por *`Antolin`* para tareas de inspección. Este dataset es el núcleo que alimenta el modelo PatchCore.

### Objetivo del dataset

  El propósito del dataset es entrenar y validar el modelo, de manera que pueda aprender el comportamiento normal de las piezas y detectar cualquier desviación como anomalía. Así, el sistema contribuye a:
*  Reducir desperdicio en fabricación.
*  Mejorar el control de calidad.
  
---

## 2. Arquitectura General
  En esta parte se detalla cómo están organizados los componentes internos del sistema y cómo interactúan entre sí.

**Componentes clave:**
- **FastAPI** → Servidor HTTP que recibe las peticiones (por ejemplo, `/predict`) y además genera automáticamente la documentación de la API (Swagger/OpenAPI).  
- **Módulo de inferencia (`main.py`)** → Núcleo del backend: recibe la imagen, la procesa, ejecuta el modelo y devuelve la respuesta JSON.  
- **Artefactos de modelo** → Ficheros que guardan el conocimiento del sistema, ubicados en `Backend/models/patchcore` (según variable `ARTIFACTS_DIR`):  
  - `memory_bank_core.npz`: contiene los embeddings de imágenes normales (la “memoria” de lo que es normal).  
  - `config.json`: guarda parámetros como el umbral (`threshold`) o la versión de embeddings.  
- **Carpeta `static/`** → Contiene recursos estáticos y las visualizaciones generadas (overlays, heatmaps, máscaras).  
- **Carpeta `templates/`** → Incluye `index.html`, un frontend minimalista que se sirve en la raíz `/`.  
- **ROI opcional** → Puede definirse mediante una máscara PNG (`ROI_PATH`) o un recorte porcentual (`IGNORE_BORDER_PCT`) para limitar la región usada en el cálculo del score.  
- **KNN precalculado** → El sistema compara cada patch de la imagen con los embeddings del *memory bank* usando un algoritmo KNN (k vecinos más cercanos) para calcular distancias de rareza.  
- **Hooks de PyTorch** → Se enganchan en capas intermedias de ResNet18 (`layer2` y `layer3`) para extraer representaciones multi-escala (texturas y formas) y fusionarlas en una representación conjunta.  

**Flujo resumido:**

```
[Cliente/Frontend] --> /predict (POST, imagen)
        |
        v
 [Lectura y normalización de imagen]
        |
        v
 [Extracción de features (ResNet18 + hooks)]
        |
        v
 [Patchify + Normalización + KNN Distancias]
        |
        v
 [Mapa de calor (heat) + Normalización (0..1)]
        |
        v
 [Aplicación ROI (opcional) para score]
        |
        v
 [Cálculo score máximo y comparación con threshold]
        |
        v
 [Generación visualizaciones (overlay/mask/polígonos) si SAVE_VIS]
        |
        v
 [Respuesta JSON (score, threshold, is_anomaly, polygons, overlay_url)]
```

---

## 3. Estructura de Directorios
  Aquí se explica cómo está distribuido el código fuente dentro del proyecto. Cada carpeta y archivo tiene un propósito claro dentro del backend (carga de modelos, visualización, templates, etc.).

```
Backend/
 ├─ main.py                # Núcleo FastAPI + lógica de inferencia
 ├─ requirements.txt       # Dependencias del entorno
 ├─ models/                # Artefactos (memory bank, config.json, etc.)
 │   └─ patchcore/         # Subcarpeta esperada (por defecto)
 ├─ static/                # Recursos estáticos + overlays generados (/static/overlays)
 ├─ templates/             # index.html para servir en "/"
 └─ tests/                 # Tests (estructura para expandir)
```


**Relación:**
- `main.py` monta `/static` y sirve `templates/index.html` en el root `/`.  
- El memory bank se carga desde `models/patchcore/memory_bank_core.npz`.  
- Visualizaciones se guardan en `static/overlays/`.  

Esta organización modular facilita el mantenimiento, la reproducibilidad y la integración con frontend y despliegues en producción.

---

## 4. Configuración y Variables de Entorno
  En este apartado se muestran las distintas variables que permiten adaptar el comportamiento del `backend` sin necesidad de modificar el código. Estas configuraciones controlan aspectos como la sensibilidad del modelo, la generación de visualizaciones, el uso de máscaras ROI y los límites de procesamiento.

**Variables (con valores por defecto si no existen en `.env`):**
- `ARTIFACTS_DIR` (por defecto: `models/patchcore`)
- `STATIC_DIR` (por defecto: `static`)
- `OVERLAYS_SUBDIR` (por defecto: `overlays`)
- `THRESHOLD` (por defecto: `config.json.threshold` o 0.35 si no está definido)
- `IMG_SIZE` (por defecto: 256)
- `KNN_K` (por defecto: 3)
- `PATCH_STRIDE` (por defecto: 1)
- `SAVE_VIS` (`"1"` habilita visualizaciones; `"0"` solo devuelve JSON)  
- `AREA_MIN` (por defecto: 200, área mínima de contornos)
- `IGNORE_BORDER_PCT` (porcentaje recortado simétricamente de cada borde para ROI)
- `ROI_PATH` (ruta a PNG binaria para máscara ROI, tamaño igual a `IMG_SIZE`)  

**Resumen de impacto:**
- Ajustan la sensibilidad y coste computacional.  
- Permiten activar o desactivar visualizaciones.  
- Controlan qué parte de la imagen entra en el cálculo del score (ROI).  
- Permiten modificar umbrales sin cambiar código. 

**Ejemplo `.env`:**
```
THRESHOLD=0.42
IMG_SIZE=256
KNN_K=5
SAVE_VIS=1
IGNORE_BORDER_PCT=8
ROI_PATH=./models/roi_mask.png
```

Estas variables permiten ajustar el sistema de forma flexible y reproducible, sin necesidad de modificar el código fuente.

---

## 5. Flujo de Inferencia Detallado
  Esta sección profundiza en todo el recorrido que sigue una imagen dentro del sistema de inferencia. Desde que se recibe y se transforma, hasta la obtención del mapa de anomalías y el resultado final. Se explican paso a paso los cálculos y operaciones que permiten al backend decidir si una imagen presenta o no una anomalía.

**Pasos:**
1. **Carga y validación de archivo**: recepción (`UploadFile`), verificación de tipo MIME/tamaño y decodificación con OpenCV (manejo BGR, conversión desde BGRA o escala de grises).  
2. **Preprocesado básico**: conversión a escala de grises y redimensionado a `IMG_SIZE`.  
3. **Tensor de entrada**: normalización a rango esperado y replicado a 3 canales (ResNet18 espera 3 canales).  
4. **Forward del backbone**: ejecución de ResNet18 con hooks en `layer2` y `layer3`.  
5. **Alineación espacial**: interpolación de `layer3` para igualar tamaño a `layer2`.  
6. **Fusión de características**: concatenación por canales `fcat = [layer2, layer3_up]`.  
7. **Patchify**: conversión de cada ubicación espacial en un vector; `stride` opcional para subsampling.  
8. **Normalización L2 por patch**: asegurar comparabilidad con la memoria.  
9. **Consulta KNN**: cada patch contra el memory bank (k vecinos).  
10. **Mapa de distancias**: distancia promedio a los `k` vecinos → mapa de rareza.  
11. **Upsampling**: redimensionado del mapa a resolución `IMG_SIZE`.  
12. **Normalización para visualización**: min-max del mapa para overlay y heatmap (no altera el cálculo del score).  
13. **Cálculo de score**: máximo del mapa dentro de la ROI (si definida); fuera de ROI se ignora/mascara.  
14. **Decisión**: comparación `score` vs `threshold` efectivo (base `.env` ajustado por `thr` y `mode`).  
15. **Visualización (opcional)**: si `SAVE_VIS=1`, generación de overlay, heatmap coloreado, máscara binaria; operaciones morfológicas y extracción/aproximación de contornos y polígonos.  
16. **Respuesta**: JSON con `score`, `threshold`, `is_anomaly`, `polygons` (si anomalía) y `overlay_url`.

```
+-------------------+
|   Imagen entrada  |
|   (UploadFile)    |
+---------+---------+
          |
          v
+-------------------+
| Preprocesado      |
| - OpenCV decode   |
| - Gris + resize   |
| - Tensor 3 canales|
+---------+---------+
          |
          v
+-------------------+
| Backbone ResNet18 |
| Hooks: layer2/l3  |
+---------+---------+
          |
          v
+-------------------+
| Alineación & Fusión|
| - Upsample layer3  |
| - Concat canales   |
+---------+---------+
          |
          v
+-------------------+
| Patchify + L2 norm|
+---------+---------+
          |
          v
+-------------------+
| KNN Memory Bank   |
| - Distancias k    |
| - Mapa rareza     |
+---------+---------+
          |
          v
+-------------------+
| Postprocesado     |
| - Upsample mapa   |
| - Min-max norm    |
| - ROI + score max |
+---------+---------+
          |
          v
+-------------------+
| Comparación       |
| score vs threshold|
| (thr/mode/.env)   |
+---------+---------+
          |
          v
+-------------------+
| Visualización     |
| - Overlay/heatmap |
| - Máscara/polígonos|
+---------+---------+
          |
          v
+-------------------+
| Respuesta JSON    |
| score, threshold, |
| is_anomaly, polys,|
| overlay_url       |
+-------------------+

```

---

## 6. Backbone y Extracción de Características
  En este punto se describe el corazón del modelo: el backbone ResNet18. Se explica cómo se aprovechan sus capas intermedias (hooks), cómo se combinan las características extraídas y por qué se usa un enfoque basado en distancias KNN sobre embeddings. El objetivo es entender cómo el sistema “aprende” a reconocer lo normal y a detectar lo que se sale de ese patrón.

>**Nota:** El backbone (ResNet18) actúa como extractor de características, generando representaciones visuales en múltiples niveles de abstracción (bordes, texturas, formas), que sirven como base para los procesos posteriores de detección de anomalías.

- **Backbone**: `ResNet18` pre-entrenado en ImageNet.  
- **Hooks**:
  - `layer2` captura características de nivel medio (textura, bordes locales).
  - `layer3` características más profundas (formas y semántica global); se aplica *upsample* para alinear espacialmente.
- **Fusión**: concatenación de canales ⇒ representación multi-escala que combina detalle fino y contexto global.
- **Normalización de patches**: asegura que los embeddings nuevos tengan distribución comparable a la almacenada en el *memory bank*, evitando escalas arbitrarias.
- **KNN**: calcula distancias promedio a los *k* vecinos más cercanos como puntaje de rareza (anomalía = embedding poco similar al banco).

**Ventajas:**
- No requiere retraining para cada clase normal (memory bank preconstruido).  
- Escalable a diferentes objetos si se reconstruye el banco.

Este diseño permite que el sistema aprenda lo normal de manera no supervisada y detecte desviaciones sin necesidad de entrenar un clasificador específico.

---

## 7. ROI y Manejo de Bordes
  Este apartado describe el manejo de las `Regiones de Interés (ROI)`. El objetivo es permitir que el sistema se concentre en áreas relevantes de la imagen, ignorando bordes o zonas irrelevantes, con el fin de reducir falsos positivos en la detección de anomalías.

**Mecanismos disponibles:**
1. **Recorte de bordes (`IGNORE_BORDER_PCT`)**: crea margen ignorado. Los píxeles en esa zona se marcan como 0 en la máscara.
2. **Máscara externa (`ROI_PATH`)**: imagen binaria (blanco = área válida). La máscara se reescala a `IMG_SIZE` y se combina con el recorte de bordes.

**Uso:**
- Al calcular el score, los píxeles fuera de ROI se penalizan (se les asigna un valor mínimo `-1`).
- En visualización, el borde de la ROI se dibuja con color cian (0,255,255).
- La máscara afecta sólo score y binarización para polígonos, no el colormap.

**Consideraciones:**
- Si la máscara final queda toda a cero, se ignora ROI (retorna `None`).
- Evita falsos positivos en áreas irrelevantes (bordes, fondo).

De esta manera, el sistema se centra únicamente en las regiones relevantes, mejorando la precisión de la detección de anomalías.

---

## 8. Cálculo del Mapa de Anomalía y Score
  Aquí se explica la lógica matemática que hay detrás del resultado. Se detalla cómo se construye el mapa de anomalía (heatmap), cómo se normalizan los valores y cómo se obtiene un “score” que resume la rareza de la imagen. También se describe cómo los modos “sensitive” y “strict” ajustan dinámicamente los umbrales.

**Definiciones:**
- `heat`: mapa en float32 resultante del resizing de distancias por patch.
- `hmin`, `hmax`: valores mínimo y máximo usados para normalización.
- `heat_norm`: escala 0..1 usada en visualizaciones y para derivar umbral relativo.
- `score`: máximo valor de `heat` dentro de la ROI (si existe), o máximo global en caso contrario.

**Interpretación:**
- Distancias mayores => más anómalo.
- Score > threshold => `is_anomaly = True`.

**Umbral efectivo:**
```
threshold_base = THRESHOLD (env o config)
if mode == "sensitive": threshold = threshold_base * 0.8
elif mode == "strict":  threshold = threshold_base * 1.2
if thr (param query) != None: threshold = thr  (override total)
```

**Normalización del umbral para máscara:**
```
thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
```
Se utiliza para segmentar el mapa normalizado en regiones normales y anómalas, facilitando la visualización y la generación de máscaras binarias.

---

## 9. Visualización y Polígonos
  Esta sección introduce la generación de los resultados visuales que ayudan a interpretar las anomalías detectadas. Se explica cómo se crean los mapas de calor, las máscaras binarias, los polígonos que delimitan zonas anómalas y cómo todo esto se guarda como archivos accesibles desde el frontend.

**Proceso en `save_visuals_and_polys`:**
1. Convierte `heat_norm` a 8 bits (0–255).
2. Genera colormap (JET).
3. Superpone colormap con la imagen gris original.
4. Binariza usando `thr_norm` si se proporcionó; en caso contrario, aplica percentil 98 de `valores > 0` como umbral adaptativo .
5. Aplica operaciones morfológicas (*open* y *close*) para eliminar ruido y cerrar huecos.
6. Extrae contornos: filtra por `AREA_MIN`.
7. Aproxima polígonos con `cv2.approxPolyDP` (epsilon fijo 2.0) para simplificar la geometría.
8. Dibuja polígonos sobre overlay.
9. Agrega borde de ROI si existe.
10. Guarda tres archivos:
   - `*_overlay.png`
   - `*_heat.png`
   - `*_mask.png`
11. Construye URLs públicas (`/static/overlays/...`).

**Control lógico:**  
- Los polígonos sólo se retornan si `is_anomaly = True`.  

**Ejemplo de respuesta parcial:**
```
{
  "score": 0.57,
  "threshold": 0.42,
  "is_anomaly": true,
  "polygons": [[[12,45],[38,44],[41,70],[10,72]]],
  "overlay_url": "/static/overlays/pieza_overlay.png"
}
```

De esta manera, el sistema no solo calcula la anomalía, sino que también ofrece una representación visual clara y accesible desde el frontend.


---

## 10. Endpoints de la API
  En esta sección se documentan los diferentes endpoints que ofrece el backend. Se explica qué hace cada uno, qué parámetros acepta, qué tipo de respuestas devuelve y cómo interactuar con ellos tanto desde un navegador como desde scripts en Python o mediante CURL.

### 10.1 GET /health
- Método: GET
- Body: ninguno
- Respuesta 200:
```
{
  "status": "ok",
  "device": "cuda" | "cpu",
  "img_size": 256,
  "knn_k": 3,
  "threshold": 0.35,
  "ignore_border_pct": 0,
  "roi_path": null | "ruta"
}
```

### 10.2 GET /
- Método: GET
- Sirve `templates/index.html` si existe; si no:
```
{ "detail": "templates/index.html no encontrado" }
```
- Uso: entregar frontend simple (subir imagen, ver overlay).

### 10.3 POST /predict
- Método: POST
- Content-Type: multipart/form-data
- Parámetros Query (opcionales):
  - `thr`: float (umbral manual)
  - `mode`: "sensitive" | "strict"
    - `sensitive` => reduce threshold 20%
    - `strict` => incrementa threshold 20%
- Campo Form:
  - `file`: imagen (jpeg/png)
- Respuestas:
  - **200 OK**:
    ```
    {
      "score": float,
      "threshold": float,
      "is_anomaly": bool,
      "polygons": [ [ [x,y], ... ], ... ],
      "overlay_url": "/static/overlays/xxx_overlay.png" | null
    }
    ```
  - **400 Bad Request**:
    - "Archivo vacío."
    - "No se pudo decodificar la imagen."
  - **500 Startup Error** (si faltan artefactos):
    - "No existe memory bank: ..." (lanzado en carga inicial)

**Ejemplo (curl):**
```
curl -X POST "http://localhost:8000/predict?mode=sensitive" \
  -F "file=@./ejemplos/pieza123.png"
```

**Ejemplo (Python requests):**
```python
import requests
with open("pieza123.png", "rb") as f:
    files = {"file": ("pieza123.png", f, "image/png")}
    r = requests.post("http://localhost:8000/predict", files=files, params={"thr":0.4})
print(r.json())
```

---

## 11. Ejemplos de Uso en Diferentes Escenarios
  Aquí se presentan ejemplos prácticos que muestran cómo utilizar la API en distintos contextos: pruebas rápidas, auditorías, o ejecuciones sin visualización. Estos ejemplos ayudan a entender mejor el uso real de los endpoints y cómo aprovechar sus parámetros

1. **Detección flexible**  
   - Ajustar el umbral dinámicamente para una tanda de imágenes con mayor ruido: usar `mode=sensitive`.
   - Para mayor rigor, se puede usar `mode=strict`, que incrementa el umbral en un 20%.
     
2. **Auditoría** 
   - Llamar `/health` para verificar que la versión del modelo cargado coincide con expectativas (umbral, IMG_SIZE).
     
3. **Visualización desactivada** 
   - Ejecutar con `SAVE_VIS=0` para reducir I/O si sólo se requiere JSON.
   - - En este modo, no se generan archivos visuales (`overlay.png`, `heat.png`, `mask.png`), y la respuesta se limita al JSON con score, threshold y polígonos.  

---

## 12. Integración con Frontend
  Esta sección explica cómo el frontend se conecta con la API. Se muestra cómo se utiliza el endpoint principal para subir imágenes y visualizar los resultados, y se proponen ideas para ampliar la interfaz (como sliders de umbral o selección de modo).

- `GET /` entrega `index.html` que actúa como un frontend mínimo de referencia. Este puede incluir:
  - Form para subir imagen.
  - Llamada `fetch` al endpoint `/predict`.
  - Renderizado del resultado mediante `overlay_url`.
- Los archivos generados (overlays, heatmaps y máscaras) se almacenan en `/static/overlays/` y pueden ser consumidos directamente por el frontend.
- Posibles ampliaciones de la interfaz:
  - Mostrar polígonos sobre un canvas interactivo.
  - Slider para ajustar el umbral (`thr`) en cliente, invocando `/predict?thr=...`.
  - Selector de modo (`sensitive` o `strict`) para modificar dinámicamente el comportamiento del detector.

De esta manera, el frontend puede ofrecer una experiencia interactiva y configurable para la detección de anomalías.

---

## 13. Dependencias (requirements.txt)
  Aquí se listan las librerías principales del proyecto y se explica brevemente el papel de cada una. También se dan recomendaciones sobre la compatibilidad y fijación de versiones, especialmente para los componentes más sensibles como PyTorch y CUDA.

**Dependencias principales:**
- fastapi: framework web para construir la API.
- uvicorn: servidor ASGI para ejecutar la aplicación.
- python-dotenv: carga de variables desde `.env`.
- numpy, opencv-python (cv2): procesamiento de imágenes y operaciones numéricas.
- torch, torchvision: backbone (ResNet18) y operaciones tensoriales.
- scikit-learn: mplementación de KNN para el memory bank.
- *(Opcional)* `pillow`: requerido por `torchvision` para algunas transformaciones de imágenes. 
- *(Opcional)* `jinja2`, `python-multipart`: útiles si se emplean templates o formularios en el frontend.

**Impactosy recomendaciones:**
- Verificar compatibilidad entre `torch` y la versión de CUDA instalada.  
- Fijar versiones en `requirements.txt` (ejemplo: `torch==2.0.1`, `fastapi==0.103.0`) para asegurar reproducibilidad del memory bank y estabilidad en despliegues.  
- Documentar dependencias opcionales según el uso real del proyecto.

La fijación de versiones asegura que el entorno sea reproducible y evita incompatibilidades en producción.

---

## 14. Artefactos del Modelo y Compatibilidad
### 14.1 Artefactos del modelo
Los artefactos son los ficheros que guardan el conocimiento del sistema y son imprescindibles para que el backend arranque correctamente.

**Ruta esperada por defecto:**
- `Backend/models/patchcore/memory_bank_core.npz`
- `Backend/models/patchcore/config.json`

Puedes cambiar la ruta con la variable de entorno `ARTIFACTS_DIR` (ej.: `ARTIFACTS_DIR=Backend/models/patchcore`).

**Opciones para disponer de los artefactos:**
- **A) Usar artefactos preconstruidos**
  1. Obtén los archivos `memory_bank_core.npz` y `config.json` desde tu fuente interna (drive/artefactos de release).
  2. Colócalos en `Backend/models/patchcore/` (o en la ruta configurada en `ARTIFACTS_DIR`).
  3. Verifica su existencia:
     - Linux/macOS: `ls -lh Backend/models/patchcore/`
     - Windows (PowerShell): `Get-ChildItem Backend/models/patchcore/`
  > Nota: Si faltan, la app fallará en el arranque con “No existe memory bank: …”.

- **B) Generar los artefactos desde imágenes normales**
  1. Reúne un dataset de imágenes normales (sin defectos), por ejemplo en `data/normal/`.
  2. Usa el mismo preprocesado que la inferencia (escala de grises, `IMG_SIZE`).
  3. Extrae embeddings con el backbone (ResNet18 con hooks en `layer2` y `layer3`), normaliza por patch (L2) y construye el banco KNN.
  4. Guarda:
     - `memory_bank_core.npz`: embeddings/matriz de memoria.
     - `config.json`: parámetros (ej. `{"threshold": 0.35, "embedding_version": "resnet18_layer2_3_concat_v1", "model_version": "1.0.0"}`).
  5. Ejemplo de comando (ajusta al script/notebook que tengas):
     ```
     python tools/build_memory_bank.py --data-dir data/normal \
       --out-dir Backend/models/patchcore --img-size 256 --k 3 --threshold 0.35
     ```
  6. Comprobación posterior: arranca el backend y consulta `GET /health` para verificar `threshold`, `img_size`, etc.

---

### 14.2 Compatibilidad (Python/CUDA)
Para asegurar reproducibilidad y rendimiento, se recomienda:

- **Versión de Python**: 3.10 o 3.11.  
- **PyTorch/torchvision**: instala versiones compatibles con tu entorno (CPU o GPU).  
  - Guía oficial: [PyTorch Get Started](https://pytorch.org/get-started/locally/)  
- **GPU (opcional)**:
  - Asegúrate de que la versión de `torch` coincide con tu versión de CUDA (ej. CUDA 12.1 ↔ build cu121).
  - Verificación rápida en Python:
    ```python
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    ```
- **Recomendación de fijado de versiones (requirements.txt):**
  - Fija `torch`, `torchvision`, `fastapi`, `uvicorn`, `opencv-python`, etc., para reproducibilidad.
  - Si usas GPU, documenta qué build de `torch` instalar (CPU-only vs cuXXX).

---

### 14.3 Ejecución de Tests con pytest
Los tests permiten validar el correcto funcionamiento del backend.

**Instalación de dependencias de test:**
- `pip install pytest`
- (Opcional) si tus tests usan clientes HTTP: `pip install httpx pytest-asyncio`

**Requisitos previos:**
- Asegúrate de que los artefactos (`memory_bank_core.npz`, `config.json`) existen en la ruta configurada.
- Para acelerar, puedes desactivar visualizaciones en tests:
  - Linux/macOS: `export SAVE_VIS=0`
  - Windows (PowerShell): `$env:SAVE_VIS="0"`

**Ejecución:**
- Ejecutar todos los tests (por ejemplo si viven en `Backend/tests`):
    - `pytest -q Backend/tests`
- Ejecutar un test concreto:
    - `pytest -q Backend/tests/test_health.py -k test_health`
 
>**Notas:**
>  - Si los tests usan FastAPI TestClient, no necesitas arrancar `uvicorn`; los tests importan la app directamente.
>  - Si un test falla con error de artefactos, revisa la ruta (`ARTIFACTS_DIR`) o coloca los archivos en `Backend/models/patchcore/`.

---

## 15. Tests
  En este apartado se explican los tipos de pruebas recomendadas para asegurar el correcto funcionamiento del backend. Se incluyen ejemplos de tests básicos (salud, predicción, errores esperados) y sugerencias para validar casos específicos como el uso de ROI o los modos de sensibilidad.

**Casos sugeridos:**
- Test de `/health` ⇒ respuesta 200 y campos esperados.  
- Test de `/predict` con imagen válida ⇒ `score` numérico y `threshold`.  
- Test de `/predict` con archivo vacío ⇒ respuesta 400.  
- Test de `/predict` con artefactos faltantes ⇒ respuesta 500 (ejemplo: memory bank inexistente).  
- Test de ROI ⇒ configurar `IGNORE_BORDER_PCT` y verificar reducción del score fuera del borde.  
- Test de `mode` ⇒ comparar resultado `is_anomaly` con y sin `mode=sensitive` dado mismo score.  
- Test de visualización ⇒ verificar que `overlay_url` apunta a `/static/overlays/...`.

  
**Ejemplo conceptual (pytest):**
```python
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "device" in data
```

---

## 16. Seguridad y Rendimiento
  Esta parte reúne las buenas prácticas para mantener el backend estable, rápido y seguro. Se comentan medidas como la validación de archivos, la limitación del tamaño de entrada, el cacheo de componentes y la configuración adecuada del CORS en entornos de producción.

**Recomendaciones:**
- **Límite de tamaño de archivo**: implementar middleware adicional para evitar cargas excesivas.  
- **Validar tipos MIME y extensiones**: asegurar que solo se acepten imágenes válidas (JPEG/PNG).  
- **Cachear backbone y KNN**: ya se realiza en *startup* para reducir latencia en inferencia.  
- **Evitar sobrescritura arbitraria de overlays**: sanear el nombre base y generar identificadores únicos (UUID/timestamp).  
- **Escalado horizontal**: usar balanceador de carga y replicar el *memory bank* en modo *read-only*.  
- **Altos volúmenes de datos**: considerar batching (no implementado) o reducir `IMG_SIZE` para optimizar rendimiento.  
- **CORS**: actualmente abierto (`allow_origins=["*"]`); en producción restringir a dominios confiables.

La aplicación de estas medidas asegura un backend robusto, eficiente y seguro en entornos de producción.
  
---

## 17. Extensiones Futuras
  Aquí se presentan ideas y líneas de mejora que podrían implementarse a futuro, como el procesamiento en lote, la autenticación, la persistencia en base de datos o la exposición de métricas de rendimiento.

**Ideas:**
- **Endpoint `GET /config`**: exponer metadatos adicionales (versión del memory bank, parámetros activos como `IMG_SIZE`, `knn_k`, `threshold`).  
- **Endpoint `POST /predict/batch`**: permitir envío de múltiples imágenes en una sola petición y devolver resultados en paralelo.  
- **Autenticación**: añadir soporte de tokens (JWT, OAuth2) para restringir el acceso a endpoints sensibles.  
- **Métricas Prometheus**: exportar indicadores de latencia, conteo de inferencias, errores y uso de recursos (CPU/GPU).  
- **Persistencia de resultados**: almacenar inferencias en una base de datos ligera (SQLite para desarrollo, PostgreSQL para producción).  
- **WebSocket**: habilitar notificaciones en tiempo real para mostrar progreso en tareas pesadas (preprocesado o batch).  

Estas extensiones permitirían escalar el sistema, mejorar la seguridad y ofrecer mayor observabilidad en entornos de producción.


---

## 18. Diagramas ASCII de Arquitectura
  En esta sección se muestran diagramas de texto que ayudan a visualizar la relación entre los distintos componentes del sistema. Son útiles para comprender de un vistazo cómo fluye la información desde el frontend hasta la inferencia y la respuesta.

### 18.1 Componentes
```
+-------------------+          +--------------------------+
|  Cliente (Web)    |  HTTP    | FastAPI (/predict,/...)  |
|  - index.html     | <------> | main.py                  |
|  - JS fetch       |          |                          |
+-------------------+          +-----------+--------------+
                                           |
                                           | Startup
                                           v
                               +----------------------------+
                               |  Backbone (ResNet18)       |
                               |  Hooks layer2 / layer3     |
                               +-------------+--------------+
                                             |
                                +------------v-------------+
                                |  Memory Bank (KNN)       |
                                |  (embeddings normal)     |
                                +------------+-------------+
                                             |
                                 +------------v-------------+
                                |  Inferencia               |
                                |  - Patchify               |
                                |  - Distancias KNN         |
                                |  - Heat / Score / ROI     |
                                |  - Comparación threshold  |
                                +------------+--------------+
                                             |
                                +------------v-------------+
                                |  Visualizaciones         |
                                |  overlays / polígonos    |
                                |  archivos en /static/... |
                                +------------+-------------+
                                             |
                                +------------v-------------+
                                |  Respuesta JSON          |
                                |  score, threshold,       |
                                |  is_anomaly, polygons,   |
                                |  overlay_url             |
                                +--------------------------+
```

---

## 20. Ejemplo Completo de Ciclo de Inferencia
  Aquí se muestra un caso práctico completo, paso a paso, de cómo el backend procesa una imagen real. Permite entender de forma concreta cómo se aplican los parámetros y cómo se interpreta la respuesta final.

Dado:
- Imagen `pieza123.png`
- `.env` con `THRESHOLD=0.40`
- Se llama: `POST /predict?mode=strict`

**Flujo:**
1. Umbral base = 0.40 → modo `strict` incrementa 20% ⇒ threshold efectivo = 0.48.  
2. Se calcula `score = 0.52`.  
3. Como `score > threshold`, se determina que la imagen es una anomalía.  
4. Se generan overlays y, tras binarización y operaciones morfológicas, se detectan dos contornos.  
5. Respuesta JSON:
```
{
  "score": 0.52,
  "threshold": 0.48,
  "is_anomaly": true,
  "polygons": [
    [[15,34],[44,33],[47,60],[13,62]],
    [[120,88],[150,87],[151,110],[119,112]]
  ],
  "overlay_url": "/static/overlays/pieza123_overlay.png"
}
```
*(Los polígonos se devuelven únicamente si `is_anomaly = true`.)*

Este ejemplo muestra cómo los parámetros de configuración y el flujo interno del backend se reflejan directamente en la respuesta final consumida por el frontend.


---

## 21. Resumen Final
  Esta última parte condensa los puntos principales de toda la documentación. Resume el propósito del backend, su flexibilidad, la capacidad de visualización y las posibles vías de ampliación para proyectos futuros.

**El backend:**
- **Core**: ofrece inferencia de anomalías eficiente con PatchCore (ResNet18 + KNN).  
- **Configuración y visualización**: es configurable vía entorno (`.env`), proporciona visualizaciones opcionales para análisis humano y permite ajustar sensibilidad por `mode` o parámetro `thr`.  
- **Integración y extensibilidad**: facilita integración con un frontend básico y es extensible hacia batching, autenticación y almacenamiento persistente.  

Este backend constituye una base sólida para proyectos de detección de anomalías y puede evolucionar hacia soluciones más complejas y escalables.


---
