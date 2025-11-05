# Backend — DP-Antolin-IA

Informe técnico funcional de la capa backend del proyecto DP-Antolin-IA. Reúne propósito, alcance, arquitectura, operación, interfaz de servicio, configuración, despliegue y consideraciones de seguridad y rendimiento.

## 1. Introducción y propósito
El backend proporciona un servicio web para analizar imágenes y detectar posibles anomalías. La detección se apoya en un modelo de visión por computador que compara las imágenes recibidas con ejemplos de referencia considerados “normales”. El servicio ofrece:
- Un resultado binario (normal/anómalo).
- Un valor numérico de rareza (score).
- Opcionalmente, una imagen con la zona sospechosa resaltada (overlay).

## 2. Alcance
- Recepción de imágenes a través de una API HTTP.
- Procesamiento y análisis de cada imagen de forma independiente.
- Respuesta estructurada en formato JSON y, si se configura, generación de recursos visuales.
- Exposición de archivos estáticos para facilitar la visualización de resultados.

## 3. Componentes principales
Objetivo: identificar los archivos relevantes del backend y explicar su función dentro del sistema.

- Backend/main.py
  - Punto de entrada de la aplicación FastAPI.
  - Configura CORS, monta archivos estáticos en `/static`, y define los endpoints.
  - Carga de configuración: variables de entorno (incluye soporte para Backend/.env) y, si existe, `Backend/models/patchcore/config.json`.
  - Inicialización en arranque (`@app.on_startup`):
    - Construcción del backbone de visión (ResNet18) para extraer información visual.
    - Carga del banco de memoria (`memory_bank_core.npz`) y preparación del KNN para comparaciones.
    - Construcción de la máscara de ROI, si se define.
  - Endpoints principales:
    - GET `/health`: estado y parámetros activos.
    - GET `/`: sirve una interfaz HTML si existe `Backend/templates/index.html`.
    - POST `/predict`: recibe una imagen, calcula score, decide anomalía y opcionalmente genera overlays.
  - Lógica de análisis (alto nivel): lectura y normalización de imagen, extracción de características, comparación con memoria (KNN), generación de mapa/score de rareza, umbralizado, postprocesado (ROI, contornos) y generación de visualizaciones.

- Backend/static/
  - Carpeta de archivos estáticos servidos por la API en `/static`.
  - Subcarpeta relevante: `Backend/static/overlays`, donde se guardan las imágenes de resultado (cuando el guardado está activado).

- Backend/static/js/app.js
  - Script de la interfaz web simple.
  - Opera contra `/health` y `/predict`, permite seleccionar umbral manual o modos (`sensitive`/`strict`), subir imágenes y visualizar resultados y overlays.

- Backend/templates/index.html (si está presente)
  - Interfaz HTML simple para ejecutar la inspección desde el navegador.
  - Si no existe, el endpoint GET `/` devuelve un mensaje indicando que la plantilla no se encontró.

- Backend/models/patchcore/
  - `config.json`: valores de configuración del modelo (por ejemplo, `threshold` recomendado).
  - `memory_bank_core.npz`: banco de memoria con referencias de normalidad. Indispensable para la inferencia.

- Backend/requirements.txt
  - Lista de dependencias Python utilizadas por el backend (FastAPI, Uvicorn, NumPy, OpenCV, PyTorch, Torchvision, scikit-learn, etc.).

- .env.example y Backend/.env
  - `.env.example`: plantilla de variables. Puede copiarse a `Backend/.env` para activar configuración por entorno.
  - `Backend/.env`: archivo leído por la aplicación para ajustar parámetros sin tocar el código (tamaño de imagen, umbral, rutas, etc.).

- Dockerfile
  - Receta de construcción de la imagen: instala dependencias del sistema y Python, copia el código, precarga pesos de ResNet18 y arranca Uvicorn escuchando en `0.0.0.0:8000`.

- docker-compose.yml
  - Orquestación para construir y ejecutar el servicio.
  - Expone el puerto 8000, monta la carpeta del modelo y la de overlays, define variables de entorno mínimas y añade un healthcheck.

- README.md y DOCKER.md
  - Documentos operativos. README ofrece guía general; DOCKER.md detalla construcción y ejecución con Docker y Docker Compose en distintos sistemas.

## 4. Arquitectura y flujo general
Objetivo: describir cómo se organiza el backend y el recorrido de una imagen desde la entrada hasta la salida.

- Organización por capas:
  - Interfaz de servicio (FastAPI): recibe peticiones, valida entradas, sirve estáticos y devuelve respuestas JSON.
  - Motor de análisis: prepara la imagen, extrae información visual, compara con referencias normales y calcula la rareza.
  - Visualización: genera (si está habilitado) overlays y los deja accesibles en `/static/overlays`.

- Flujo de inicio:
  1) Lectura de configuración desde variables de entorno, Backend/.env y, si procede, `config.json` del modelo.
  2) Construcción del backbone de visión (ResNet18) y registro de puntos de extracción.
  3) Carga del banco de memoria y preparación de KNN para consultas rápidas.
  4) Preparación de ROI (máscara o recorte de bordes).
  5) Montaje de directorio estático y verificación de rutas de salida para overlays.

- Flujo de análisis por petición:
  1) Recepción de la imagen (multipart/form-data, campo `file`).
  2) Normalización (por ejemplo, escala a `IMG_SIZE` y conversión acorde a lo esperado).
  3) Extracción de características con el backbone sin reentrenamiento.
  4) Comparación con la memoria mediante KNN para estimar rareza local.
  5) Agregación a score global y cálculo del umbral efectivo (base ± modo o `thr` manual).
  6) Postprocesado: ROI, filtrado por área mínima, contornos y polígonos.
  7) Generación y almacenamiento de overlays (si está activo), publicación bajo `/static`.
  8) Respuesta JSON con `is_anomaly`, `score`, `threshold` y, si aplica, `overlay_url`.

## 5. Interfaz de la API
Objetivo: definir la interfaz del servicio, sus endpoints, entradas y salidas.

- GET `/health`
  - Finalidad: verificación de estado y consulta de parámetros activos.
  - Respuesta: objeto JSON con campos como `status`, `device` (cpu/cuda), `img_size`, `knn_k`, `threshold`, `ignore_border_pct`, `roi_path`.

- GET `/`
  - Finalidad: servir una interfaz HTML si existe `Backend/templates/index.html`.
  - Si la plantilla no existe, devuelve un mensaje indicándolo.

- POST `/predict`
  - Finalidad: analizar una imagen y devolver el resultado.
  - Entrada: `multipart/form-data` con el campo `file` (imagen).
  - Parámetros de consulta:
    - `thr` (float, opcional): umbral manual para la petición.
    - `mode` (string, opcional): `sensitive` (umbral algo más bajo) o `strict` (umbral algo más alto).
  - Respuesta (campos habituales):
    - `is_anomaly` (boolean): decisión final.
    - `score` (float): valor de rareza calculado.
    - `threshold` (float o string): umbral usado.
    - `overlay_url` (string, opcional): ruta a la imagen superpuesta, cuando el guardado está activado.

Ejemplo de respuesta:
```json
{
  "is_anomaly": true,
  "score": 0.412345,
  "threshold": 0.356087,
  "overlay_url": "/static/overlays/ejemplo_overlay.png"
}
```

## 6. Configuración
Objetivo: explicar por qué y cómo se ajusta el backend a distintos entornos y qué parámetros influyen en el comportamiento.

- Fuentes de configuración:
  - Variables de entorno (incluye Backend/.env si existe).
  - `Backend/models/patchcore/config.json` como fuente para algunos valores (por ejemplo, `threshold` recomendado).
  - Parámetros de la petición (`thr`, `mode`) que aplican solo a esa llamada.

- Variables principales:
  - `IMG_SIZE` (entero): tamaño objetivo de análisis. Debe ser compatible con los artefactos del modelo.
  - `KNN_K` (entero): número de vecinos consultados en la comparación con la memoria (sensibilidad/tiempo).
  - `THRESHOLD` (float): umbral base para decidir anomalía.
  - `IGNORE_BORDER_PCT` (float): recorte porcentual de bordes para evitar regiones poco relevantes.
  - `ARTIFACTS_DIR` (ruta): carpeta con `memory_bank_core.npz` y `config.json`.
  - `STATIC_DIR` (ruta) y `OVERLAYS_SUBDIR` (texto): ubicación donde se publican y guardan overlays.
  - `ROI_PATH` (ruta, opcional): máscara binaria (blanco/negro) que delimita el área evaluada.
  - `SAVE_VIS` (0/1, opcional): activa el guardado de overlays.
  - Otros potenciales: `PATCH_STRIDE`, `AREA_MIN`.

- Recomendación operativa:
  - Empezar con valores de `.env.example`.
  - Ajustar el umbral (`THRESHOLD` o `mode`) según la sensibilidad buscada.
  - Emplear ROI para centrar el análisis en zonas relevantes y reducir falsas alarmas.

## 7. Datos y artefactos del modelo
Objetivo: detallar los ficheros que habilitan la inferencia y su coherencia con la configuración.

- Banco de memoria: `Backend/models/patchcore/memory_bank_core.npz`.
  - Contiene referencias de normalidad necesarias para el cálculo de rareza.
  - Su ausencia impide realizar la inferencia.

- Configuración del modelo: `Backend/models/patchcore/config.json`.
  - Puede contener `threshold` recomendado, ajustado al banco de memoria disponible.

- Coherencia:
  - `IMG_SIZE` y otros parámetros deben corresponderse con los usados para generar la memoria del modelo.
  - Cambios no alineados degradan el resultado.

## 8. Resultados y visualizaciones
Objetivo: especificar dónde se guarda la salida visual y cómo se accede.

- Overlays y heatmaps se guardan en `STATIC_DIR/OVERLAYS_SUBDIR` (por defecto, `Backend/static/overlays`) cuando `SAVE_VIS=1`.
- Acceso desde navegador: `/static/...`. La API devuelve `overlay_url` para enlazar el resultado.

## 9. Ejecución y despliegue
Objetivo: ofrecer instrucciones explícitas para ejecutar el backend localmente o en contenedor.

- Ejecución local (entorno Python):
  1) Instalar dependencias del sistema si fuese necesario (OpenCV, etc.).
  2) Crear entorno virtual e instalar Python deps:
     ```
     cd Backend
     python -m venv .venv
     # Windows PowerShell: .\.venv\Scripts\Activate.ps1
     # Linux/macOS/WSL2: source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
  3) Copiar variables (opcional):
     ```
     cp ../.env.example .env
     ```
  4) Verificar artefactos del modelo:
     ```
     ls models/patchcore/memory_bank_core.npz
     ```
  5) Arrancar el servidor:
     ```
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```

- Ejecución con Docker Compose (recomendado para entorno homogéneo):
  1) Desde la raíz del repo:
     ```
     docker compose up --build
     ```
  2) Verificación de servicio (UI y salud):
     - Interfaz: http://localhost:8000 o http://12.0.0.1:8000
     - Salud: http://localhos:8000/health

- Ejecución con Docker (build + run):
  1) Construir imagen:
     ```
     docker build -t inspector-patchcore:latest .
     ```
  2) Ejecutar contenedor:
     ```
     docker run --rm -p 8000:8000 \
       -e ARTIFACTS_DIR=/app/Backend/models/patchcore \
       -e STATIC_DIR=/app/Backend/static \
       -e OVERLAYS_SUBDIR=overlays \
       -v "$(pwd)"/Backend/models/patchcore:/app/Backend/models/patchcore:ro \
       -v "$(pwd)"/Backend/static/overlays:/app/Backend/static/overlays \
       inspector-patchcore:latest
     ```
  3) Verificación de servicio:
     - Interfaz: http://127.0.0.1:8000 o http://localhost:8000
     - Salud: http://127.0.0.1:8000/health

## 10. Seguridad y permisos básicos
- CORS configurado para aceptar orígenes abiertos en desarrollo; recomendable restringir orígenes en producción.
- Control de tamaño y tipo de archivos subidos para prevenir usos indebidos.
- Permisos de archivos y rutas: asegurar acceso de lectura al banco de memoria y escritura (si aplica) a la carpeta de overlays.

## 11. Rendimiento
- Uso de GPU si está disponible; en caso contrario, CPU.
- Parámetros con impacto: `IMG_SIZE`, `KNN_K`, `PATCH_STRIDE`, ROI (`IGNORE_BORDER_PCT`, `ROI_PATH`).
- La primera petición puede incluir latencia adicional por carga de pesos y estructuras internas.

## 12. Errores comunes y diagnóstico
- “No existe memory bank: …/memory_bank_core.npz”:
  - Falta el archivo o no está montado en contenedor. Solución: `git lfs pull`, verificar ruta y montaje.
- UI no visible en `/`:
  - Falta `Backend/templates/index.html`. Usar directamente `/health` y `/predict` o añadir la plantilla.
- `overlay_url` no abre:
  - Verificar `SAVE_VIS=1`, existencia del archivo en `Backend/static/overlays` y que el volumen esté correctamente montado en Docker.
- Respuestas inesperadas en bordes:
  - Ajustar `IGNORE_BORDER_PCT` o definir `ROI_PATH` para limitar el área analizada.
- Puerto ocupado:
  - Cambiar el mapeo de puertos (por ejemplo, `-p 8001:8000`) y usar `http://127.0.0.1:8001/health`.

## 13. Glosario
- Score: valor que mide el grado de rareza de una imagen (a mayor valor, más raro).
- Umbral (threshold): límite a partir del cual se clasifica como anómala.
- Overlay: imagen con marcas que señalan las zonas sospechosas.
- ROI (Región de Interés): área de la imagen sobre la que se realiza el análisis.
- Banco de memoria (memory bank): referencias de normalidad usadas para comparar.
- Backbone: red de visión utilizada para extraer información de las imágenes.
