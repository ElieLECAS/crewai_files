import os
import glob
import yaml
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process, LLM

# Import des outils personnalisés
from src.tools.pdf_reader_tool import read_pdf_file
from src.tools.json_file_writer_tool import create_json_file_writer

# 1. Chargement de l'environnement
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "mistral:7b")
TIMEOUT = int(os.getenv("TIMEOUT", "600"))

# Chemins des volumes Docker
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
CONFIG_DIR = "/app/config"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_config(filename, key):
    """Charge la configuration depuis les fichiers YAML."""
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get(key, {})

def main():
    print(f"Démarrage de l'extraction avec {MODEL}...")

    # 2. Configuration du LLM natif pour Ollama
    native_llm = LLM(
        model=f"ollama/{MODEL}",
        base_url=OLLAMA_BASE_URL,
        timeout=TIMEOUT,
        temperature=0.1,  # Température basse pour des réponses stables et déterministes
        max_tokens=4000   # Contexte suffisant pour garantir la complétude des réponses
    )

    # 3. Initialisation des outils
    # Outil personnalisé qui accepte des objets JSON et les convertit automatiquement
    json_file_writer = create_json_file_writer(root_dir=OUTPUT_DIR)
    
    # On charge les configurations
    agent_config = load_config("agents.yaml", "devis_extractor")
    read_task_config = load_config("tasks.yaml", "read_pdf_and_extract_data")
    write_task_config = load_config("tasks.yaml", "write_json_file")

    # 4. Création de l'Agent (peut utiliser les deux outils selon la tâche)
    extractor_agent = Agent(
        role=agent_config.get('role'),
        goal=agent_config.get('goal'),
        backstory=agent_config.get('backstory'),
        llm=native_llm,
        tools=[read_pdf_file, json_file_writer], 
        verbose=agent_config.get('verbose', True), 
        allow_delegation=agent_config.get('allow_delegation', False),
        max_iter=agent_config.get('max_iter', 5),
        max_execution_time=agent_config.get('max_execution_time', 300),
        memory=agent_config.get('memory', False)
    )

    # 6. Traitement des fichiers PDF
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    pdf_files.sort()
    
    if not pdf_files:
        print(f"❌ Aucun fichier PDF trouvé dans {INPUT_DIR}")
        return

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        output_name = pdf_name.replace(".pdf", ".json")
        
        print(f"\n🚀 Analyse de : {pdf_name}")

        # Tâche 1 : Lire le PDF et extraire les données
        read_task_description = (
            f"{read_task_config.get('description')}\n\n"
            f"Fichier PDF à traiter : {pdf_path}\n"
            f"Utilisez l'outil 'read_pdf_file' avec file_path='{pdf_path}' "
            f"(passer directement la valeur comme chaîne, pas un objet JSON Schema)."
        )

        read_task = Task(
            description=read_task_description,
            expected_output=read_task_config.get('expected_output'),
            agent=extractor_agent
        )

        # Tâche 2 : Sauvegarder les données dans un fichier JSON
        write_task_description = (
            f"{write_task_config.get('description')}\n\n"
            f"Nom du fichier de sortie : {output_name}\n"
            f"Utilisez l'outil 'json_file_writer' avec :\n"
            f"- filename='{output_name}'\n"
            f"- content=<les_données_extraites_de_la_tâche_précédente>\n"
            f"- directory='./'\n"
            f"Les données à sauvegarder sont le résultat de la tâche précédente."
        )

        write_task = Task(
            description=write_task_description,
            expected_output=write_task_config.get('expected_output'),
            agent=extractor_agent,
            context=[read_task]  # La tâche d'écriture dépend de la tâche de lecture
        )

        crew = Crew(
            agents=[extractor_agent],
            tasks=[read_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()

if __name__ == "__main__":
    main()