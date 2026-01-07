"""Outil personnalisé pour lire et extraire le texte des fichiers PDF."""
from langchain.tools import tool
import pdfplumber
import os


@tool
def read_pdf_file(file_path: str) -> str:
    """
    Lit un fichier PDF et extrait tout son contenu textuel.
    
    Args:
        file_path: Le chemin complet vers le fichier PDF à lire.
    
    Returns:
        Le texte extrait du PDF, organisé par pages.
    """
    try:
        if not os.path.exists(file_path):
            return f"Erreur: Le fichier {file_path} n'existe pas."
        
        if not file_path.lower().endswith('.pdf'):
            return f"Erreur: {file_path} n'est pas un fichier PDF."
        
        text_content = []
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


# Export de l'outil
PDFReaderTool = read_pdf_file

