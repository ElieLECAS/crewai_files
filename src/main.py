"""
Script principal pour l'extraction de données de devis PDF vers JSON.
Utilise CrewAI avec Ollama (Mistral 7B).
"""
import os
import json
import glob
import yaml
import re
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process, LLM
import pdfplumber

# [cite_start]1. Chargement de l'environnement et des variables [cite: 1, 2]
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "mistral:7b")
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
TIMEOUT = int(os.getenv("TIMEOUT", "600"))

# [cite_start]Chemins de configuration et de données [cite: 3]
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
CONFIG_DIR = "/app/config"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_config(filename, key):
    """Charge une configuration depuis les fichiers YAML dans /config."""
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get(key, {})

def extract_pdf_text(pdf_path):
    """Extrait le texte brut du PDF pour l'analyse par l'IA."""
    try:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        return f"Erreur lecture PDF: {str(e)}"

def extract_json_from_output(output_text):
    """Nettoie et extrait l'objet JSON de la réponse de l'Agent."""
    text = str(output_text)
    # Supprime les balises markdown si présentes
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    
    # Capture le contenu entre les premières et dernières accolades
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def validate_and_fix_data(data):
    """Valide et recalcule les totaux pour garantir la précision mathématique."""
    try:
        # Recalcule le montant HT de chaque ligne (quantité * prix_unitaire)
        for item in data.get('prestations', []):
            item['montant_ht'] = round(float(item['quantite']) * float(item['prix_unitaire']), 2)
        
        # Recalcule les totaux globaux
        total_ht = sum(item['montant_ht'] for item in data.get('prestations', []))
        data['totaux']['total_ht'] = round(total_ht, 2)
        data['totaux']['tva'] = round(total_ht * 0.20, 2)
        data['totaux']['total_ttc'] = round(data['totaux']['total_ht'] + data['totaux']['tva'], 2)
    except (KeyError, ValueError, TypeError):
        pass # Garde les données telles quelles si la structure est imprévue
    return data

def main():
    print("="*60)
    print(f"Extraction PDF -> JSON | Modèle: {MODEL}")
    print("="*60)

    # [cite_start]Configuration du LLM natif pour Ollama (évite l'appel à OpenAI) [cite: 2]
    native_llm = LLM(
        model=f"ollama/{MODEL}",
        base_url=OLLAMA_BASE_URL,
        timeout=TIMEOUT
    )

    # [cite_start]Chargement des configurations Agent et Task [cite: 2]
    agent_config = load_config("agents.yaml", "devis_extractor")
    task_config = load_config("tasks.yaml", "extract_devis_data")

    # Initialisation de l'Agent sans outils pour maximiser la concentration du LLM
    extractor_agent = Agent(
        role=agent_config.get('role'),
        goal=agent_config.get('goal'),
        backstory=agent_config.get('backstory'),
        llm=native_llm,
        verbose=VERBOSE,
        allow_delegation=False
    )

    # Récupération des fichiers PDF à traiter
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ Aucun fichier trouvé dans {INPUT_DIR}")
        return

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n🚀 Traitement de : {pdf_name}")
        
        # Extraction du texte du PDF
        pdf_text = extract_pdf_text(pdf_path)
        
        # Création de la tâche spécifique
        extraction_task = Task(
            description=f"{task_config.get('description')}\n\nCONTENU DU PDF :\n{pdf_text}",
            expected_output=task_config.get('expected_output'),
            agent=extractor_agent
        )

        # Exécution par CrewAI
        crew = Crew(
            agents=[extractor_agent],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=VERBOSE
        )

        result = crew.kickoff()
        
        # Post-traitement et sauvegarde
        json_data = extract_json_from_output(result)
        if json_data:
            json_data["fichier_source"] = pdf_name
            json_data = validate_and_fix_data(json_data) # Correction mathématique
            
            output_path = os.path.join(OUTPUT_DIR, pdf_name.replace('.pdf', '.json'))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Fichier généré : {os.path.basename(output_path)}")
        else:
            print(f"⚠️ Échec du parsing JSON pour {pdf_name}")

    print("\n" + "="*60)
    print("Traitement terminé.")
    print("="*60)

if __name__ == "__main__":
    main()