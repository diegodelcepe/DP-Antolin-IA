Clonar y cambiar a la rama docker-mvp
# Clonar
git clone https://github.com/diegodelcepe/DP-Antolin-IA.git

cd DP-Antolin-IA

# Cambiar a la rama con Docker
git fetch origin && git checkout docker-mvp

docker compose build --no-cache
docker compose up -d
docker compose logs -f
