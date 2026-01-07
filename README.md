# CrewAI - Traitement de PDFs de Devis

Ce projet utilise CrewAI pour extraire automatiquement les informations des PDFs de devis et les convertir en JSON structuré.

## 🚀 Fonctionnalités

- **Agent CrewAI** spécialisé dans l'extraction de données de devis
- **Génération automatique** de 10 PDFs de test au démarrage
- **Traitement automatique** de tous les PDFs dans le dossier `input/`
- **Sortie JSON** structurée dans le dossier `output/`
- **Intégration Ollama** pour le LLM (modèle local, pas besoin d'API externe)

## 📋 Prérequis

- Docker et Docker Compose installés
- Aucune clé API nécessaire (Ollama fonctionne localement)

## 🛠️ Installation

1. **Cloner ou télécharger le projet**

2. **Configurer les variables d'environnement (optionnel)**
   ```bash
   cp .env_example .env
   ```
   
   Par défaut, le système utilise `llama3.2:3b`. Vous pouvez modifier le modèle dans `.env` :
   ```
   MODEL=llama3.2:3b
   OLLAMA_BASE_URL=http://ollama:11434
   ```

3. **Lancer avec Docker Compose**
   ```bash
   docker-compose up --build
   ```
   
   Le premier démarrage peut prendre du temps car Ollama télécharge automatiquement le modèle `llama3.2:3b` (environ 2GB).

## 📁 Structure du projet

```
crewai_files/
├── docker-compose.yml      # Configuration Docker Compose
├── Dockerfile              # Image Docker
├── requirements.txt        # Dépendances Python
├── .env_example           # Exemple de fichier d'environnement
├── .env                   # Variables d'environnement (à créer)
├── input/                 # Dossier pour les PDFs à traiter
├── output/                # Dossier pour les JSONs générés
└── src/
    ├── main.py            # Point d'entrée principal
    ├── pdf_agent.py       # Agent CrewAI
    ├── pdf_processor_tool.py  # Tools pour CrewAI
    ├── generate_test_pdfs.py  # Script de génération de PDFs de test
    └── example_output.json    # Exemple de JSON attendu
```

## 🔧 Utilisation

### Traitement automatique

Le conteneur traite automatiquement tous les PDFs présents dans le dossier `input/` au démarrage.

### Ajouter vos propres PDFs

1. Placez vos fichiers PDF dans le dossier `input/`
2. Relancez le conteneur :
   ```bash
   docker-compose restart
   ```

### Consulter les résultats

Les JSONs générés sont disponibles dans le dossier `output/`, avec le même nom que le PDF source mais avec l'extension `.json`.

## 📊 Format JSON de sortie

Le JSON généré suit cette structure :

```json
{
  "numero_devis": "DEV-0001",
  "date_emission": "15/01/2024",
  "date_validite": "15/02/2024",
  "entreprise": {
    "nom": "Nom de l'entreprise",
    "adresse": "Adresse complète"
  },
  "prestations": [
    {
      "description": "Description de la prestation",
      "quantite": 1,
      "prix_unitaire": 1000,
      "montant_ht": 1000
    }
  ],
  "totaux": {
    "total_ht": 1000,
    "tva": 200,
    "total_ttc": 1200
  },
  "conditions_paiement": {
    "delai": "30 jours",
    "mode": "Virement bancaire"
  },
  "fichier_source": "nom_du_fichier.pdf"
}
```

## 🧪 PDFs de test

Le script `generate_test_pdfs.py` génère automatiquement 10 PDFs de devis fictifs au démarrage du conteneur pour tester le système.

## 🔍 Agent CrewAI

L'agent est configuré avec :
- **Role**: Expert en Extraction de Données de Devis
- **Tools**: Extraction PDF, Lecture de fichiers, Sauvegarde JSON
- **Process**: Séquentiel (un PDF à la fois)
- **LLM**: Ollama avec le modèle `llama3.2:3b` (exécuté localement dans Docker)

## 📝 Notes

- Les PDFs de test sont générés uniquement si le dossier `input/` est vide ou contient moins de 10 PDFs
- Le traitement est séquentiel pour garantir l'isolation du contexte entre chaque PDF
- Les erreurs sont loggées mais n'arrêtent pas le traitement des autres fichiers
- Ollama fonctionne entièrement en local, aucune connexion internet n'est nécessaire après le téléchargement du modèle
- Le modèle `llama3.2:3b` est plus léger que GPT-4 mais peut être moins précis sur des tâches complexes

## 🐛 Dépannage

### Port 11434 déjà utilisé
Si vous voyez l'erreur "address already in use" sur le port 11434 :
1. Arrêtez toute instance Ollama existante : `docker stop ollama_server` ou `pkill ollama`
2. Ou modifiez `docker-compose.yml` pour utiliser un port différent
3. Par défaut, le port n'est pas exposé à l'hôte (communication interne Docker uniquement)

### Ollama n'est pas accessible
Si vous voyez une erreur de connexion à Ollama :
1. Vérifiez que le service Ollama est démarré : `docker-compose ps`
2. Attendez quelques secondes après le démarrage pour qu'Ollama soit prêt
3. Vérifiez les logs : `docker-compose logs ollama`

### Modèle non trouvé
Si le modèle `llama3.2:3b` n'est pas disponible :
1. Le système tentera de le télécharger automatiquement au premier démarrage
2. Vous pouvez aussi le télécharger manuellement :
   ```bash
   docker exec ollama_server ollama pull llama3.2:3b
   ```

### Aucun PDF traité
Vérifiez que des fichiers PDF sont bien présents dans le dossier `input/`.

### Erreurs de traitement
Consultez les logs Docker pour voir les détails des erreurs :
```bash
docker-compose logs -f crewai
```

### Changer de modèle
Pour utiliser un autre modèle Ollama :
1. Modifiez `MODEL` dans votre fichier `.env`
2. Assurez-vous que le modèle est téléchargé dans Ollama :
   ```bash
   docker exec ollama_server ollama pull <nom-du-modele>
   ```

