FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système pour Docling
# poppler-utils : Souvent utile pour la manipulation de PDF
# libmagic1 : Utile pour la détection de formats
# libgl1 : Bibliothèque OpenGL requise par Docling pour le traitement d'images
# libglib2.0-0 : Bibliothèque GLib requise (contient libgthread-2.0.so.0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Installation de PyTorch CPU avant Docling pour éviter la version GPU (CUDA) volumineuse
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet
COPY . .

# Création des dossiers pour les volumes
RUN mkdir -p input output

CMD ["python", "src/main.py"]