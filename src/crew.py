import os
import yaml
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from src.tools.pdf_tools import PDFTools
from pydantic import BaseModel, Field
from typing import List

# --- Schémas Pydantic pour la sortie structurée ---
class Prestation(BaseModel):
    description: str
    quantite: int
    prix_unitaire: float
    montant_ht: float

class Entreprise(BaseModel):
    nom: str
    adresse: str

class Totaux(BaseModel):
    total_ht: float
    tva: float
    total_ttc: float

class DevisSortie(BaseModel):
    numero_devis: str
    date_emission: str
    date_validite: str
    entreprise: Entreprise
    prestations: List[Prestation]
    totaux: Totaux
    fichier_source: str

# --- Classe Principale du Crew ---
class InvoiceProcessingCrew:
    def __init__(self):
        # Configuration du LLM Ollama
        self.llm = Ollama(
            model=os.getenv("MODEL", "llama3.2:1b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            num_ctx=4096,        # Limite la fenêtre de contexte pour économiser la RAM
            temperature=0.1,     # Quasi-déterministe pour l'extraction de données
            repeat_penalty=1.2
        )
        
        # Chargement des configurations YAML
        with open('config/agents.yaml', 'r', encoding='utf-8') as f:
            self.agents_config = yaml.safe_load(f)
        with open('config/tasks.yaml', 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)

    def pdf_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['pdf_analyst'],
            tools=[PDFTools.extract_pdf_content],
            llm=self.llm,
            verbose=True,
            max_iter=1,  # Plus d'itérations pour permettre une meilleure extraction
            max_rpm=10,
            max_execution_time=120,  # Plus de temps pour traiter le document
            allow_delegation=False,
            memory=False,
            cache=False
        )

    def extraction_task(self, analyst_agent: Agent) -> Task:
        # On récupère la config mais on retire la clé 'agent' qui est un string 
        # pour éviter le conflit avec l'objet Agent réel
        task_config = self.tasks_config['extraction_task'].copy()
        if 'agent' in task_config:
            del task_config['agent']

        return Task(
            config=task_config,
            agent=analyst_agent,
            output_pydantic=DevisSortie
        )

    def kickoff(self, pdf_path: str, pdf_filename: str):
        analyst = self.pdf_analyst()
        task = self.extraction_task(analyst)
        
        crew = Crew(
            agents=[analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        # On s'assure que pdf_path est le chemin COMPLET (/app/input/devis_...)
        full_path = os.path.abspath(pdf_path) 
        
        result = crew.kickoff(inputs={
            "pdf_path": full_path,
            "pdf_filename": pdf_filename
        })
        
        # Extraction de l'objet Pydantic du résultat
        # CrewAI retourne le résultat dans tasks_output[0].raw pour les tâches avec output_pydantic
        if hasattr(result, 'tasks_output') and result.tasks_output:
            task_output = result.tasks_output[0]
            if hasattr(task_output, 'raw'):
                return task_output.raw
        
        # Fallback: retourner le résultat tel quel
        return result