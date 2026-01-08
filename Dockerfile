FROM python:3.11-slim

WORKDIR /app

# Installation minimale des dépendances système
# poppler-utils pour pdf2image (conversion PDF -> images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet
COPY . .

# Création des dossiers pour les volumes
RUN mkdir -p input output

CMD ["python", "src/main.py"]