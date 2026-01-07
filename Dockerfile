FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# IMPORTANT : Copier les dossiers src ET config
COPY . .

# Création des dossiers pour les volumes
RUN mkdir -p input output

CMD ["python", "src/main.py"]