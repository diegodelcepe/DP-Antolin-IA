# Inspector de Anomalías (PatchCore)

Backend con FastAPI + Frontend estático para detectar anomalías con PatchCore.

## Requisitos
- Python 3.10+
- (Opcional) CUDA si usas GPU
- Git LFS (ya configurado en el repo)

## Setup rápido
```bash
git clone https://github.com/DelcastApe/Inspector-anomalias-patchcore.git
cd Inspector-anomalias-patchcore/Backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env  # ajusta variables si quieres
uvicorn main:app --reload --port 8000
```

## Despliegue con Docker Compose
El sistema puede ejecutarse en cualquier equipo con **Docker Desktop (WSL2)** sin necesidad de instalar Python ni dependencias manuales.

### Construcción y ejecución
```bash
docker compose up --build
```
