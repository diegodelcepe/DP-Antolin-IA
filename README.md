# Inspector de Anomalías (PatchCore)

Backend con FastAPI + Frontend estático para detectar anomalías con PatchCore.
Este proyecto forma parte del reto propuesto por Antolín, dentro del programa FaCyL Talent Toolkit y del curso de Gestión de Proyectos (Universidad de León).

## Descripción del proyecto
El sistema detecta defectos visuales en piezas plásticas analizando imágenes del proceso de fabricación. Utiliza el modelo PatchCore, que aprende la apariencia normal de las piezas sin necesidad de ejemplos de fallos, y señala cualquier desviación como anomalía.

## Tecnologías utilizadas
 - Backend: FastAPI
 - Frontend: HTML + CSS + JavaScript
 - Modelo IA: PatchCore (implementado en PyTorch)
 - Lenguaje: Python 3.10+
 - Contenedores: Docker (opcional, para despliegue portable)
 - Dataset: Imágenes proporcionadas por Antolín

## Requisitos
- Python 3.10+
- (Opcional) CUDA si usas GPU
- Git LFS (ya configurado en el repo)

## Instalación y ejecución rápida
```bash
# Clonar el repositorio
git clone https://github.com/DelcastApe/Inspector-anomalias-patchcore.git
cd Inspector-anomalias-patchcore/Backend

# Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txtç

# Configurar variables de entorno
copy .env.example .env  # ajusta variables si quieres

# Ejecutar el servidor
uvicorn main:app --reload --port 8000
```
Luego abre en tu navegador: http://localhost:8000

## Estructura del proyecto
```
Backend/
│
├── models/
│   ├── patchcore/              # Implementación del modelo PatchCore
│   ├── config.json             # Configuración de parámetros del modelo
│   └── memory_bank_core...     # Memoria de características aprendidas
│
├── static/
│   ├── css/app.css             # Estilos de la interfaz
│   ├── js/app.js               # Lógica del frontend
│   └── overlays/               # Imágenes o máscaras de resultados
│
├── templates/
│   └── index.html              # Página principal del inspector
│
├── main.py                     # Servidor FastAPI
├── requirements.txt            # Dependencias de Python
└── .env                        # Variables de entorno 
```

## Autores 
Proyecto desarrollado por estudiantes del Grado en Ingeniería Informática (Universidad de León)
en colaboración con FaCyL y Antolín dentro del programa Talent Toolkit. 

## Licencia y uso
El código se distribuye bajo licencia académica.
El dataset pertenece a Antolín y su uso está restringido a fines educativos y no comerciales.
