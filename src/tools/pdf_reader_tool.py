from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator
from typing import Type, Any
import pdfplumber
import os


class ReadPdfFileInput(BaseModel):
    """Schéma d'entrée pour l'outil de lecture PDF."""
    file_path: str = Field(..., description="Le chemin complet vers le fichier PDF à lire")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalise les données d'entrée pour gérer différents formats."""
        if isinstance(data, dict):
            # Si le format est {"properties": {"file_path": "..."}}, extraire file_path
            if 'properties' in data and isinstance(data['properties'], dict):
                if 'file_path' in data['properties']:
                    return {'file_path': data['properties']['file_path']}
            # Si file_path est déjà au niveau racine, le retourner tel quel
            if 'file_path' in data:
                return data
        return data


class ReadPdfFileTool(BaseTool):
    name: str = "read_pdf_file"
    description: str = (
        "Lit un fichier PDF binaire et extrait tout son contenu textuel. "
        "Utile pour analyser des devis ou factures sans erreurs d'encodage. "
        "L'argument file_path doit être passé comme une chaîne de caractères simple."
    )
    args_schema: Type[BaseModel] = ReadPdfFileInput

    def _run(self, file_path: str) -> str:
        """
        Exécute la lecture du fichier PDF.
        
        Args:
            file_path: Le chemin complet vers le fichier PDF à lire
        
        Returns:
            Le contenu textuel extrait du PDF, organisé par pages.
        """
        try:
            if not os.path.exists(file_path):
                return f"Erreur: Le fichier {file_path} n'existe pas."
            
            text_content = []
            # pdfplumber gère l'ouverture binaire automatiquement
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Page {page_num} ---\n{page_text}\n")
            
            if not text_content:
                return f"Aucun texte n'a pu être extrait du fichier {file_path}."
            
            return "\n".join(text_content)
        
        except Exception as e:
            return f"Erreur lors de la lecture du PDF {file_path}: {str(e)}"


# Créer une instance de l'outil pour l'export
read_pdf_file = ReadPdfFileTool()