from crewai_tools import tool
import pdfplumber
import os

class PDFTools:
    @tool("extract_pdf_content")
    def extract_pdf_content(pdf_path: str) -> str:
        """
        Extrait le texte brut d'un PDF de manière fiable.
        
        Args:
            pdf_path: Chemin vers le fichier PDF (peut être un dict avec clé 'pdf_path' ou un string)
        
        Returns:
            Le texte extrait du PDF, page par page, séparé par des retours à la ligne.
            En cas d'erreur, retourne un message d'erreur descriptif.
        """
        # Gestion de différents formats d'input
        if isinstance(pdf_path, dict):
            path = pdf_path.get("pdf_path", pdf_path.get("pdf_filename", None))
            if path is None:
                return "Erreur : format de chemin PDF invalide (dict sans clé 'pdf_path')"
        else:
            path = str(pdf_path)
        
        # Normalisation du chemin
        if not os.path.isabs(path):
            # Si chemin relatif, essayer de le construire
            if path.startswith("input/") or path.startswith("./input/"):
                path = os.path.abspath(path)
            else:
                # Essayer le chemin standard /app/input/
                path = f"/app/input/{os.path.basename(path)}"
        elif not path.startswith("/app/input/") and not os.path.exists(path):
            # Si le chemin absolu n'existe pas, essayer /app/input/
            basename = os.path.basename(path)
            alt_path = f"/app/input/{basename}"
            if os.path.exists(alt_path):
                path = alt_path

        # Vérification de l'existence du fichier
        if not os.path.exists(path):
            return f"Erreur : Le fichier PDF n'existe pas à l'emplacement : {path}"

        try:
            # Extraction du texte avec pdfplumber
            full_text = []
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Ajout d'un séparateur de page pour faciliter la lecture
                        if page_num > 1:
                            full_text.append(f"\n--- Page {page_num} ---\n")
                        full_text.append(page_text)
            
            extracted_text = "".join(full_text)
            
            if not extracted_text.strip():
                return f"Avertissement : Le PDF {path} semble vide ou ne contient pas de texte extractible (peut-être un scan sans OCR)."
            
            return extracted_text
            
        except pdfplumber.exceptions.PDFSyntaxError as e:
            return f"Erreur de syntaxe PDF : {str(e)}. Le fichier {path} pourrait être corrompu."
        except Exception as e:
            return f"Erreur lors de l'extraction du PDF {path} : {str(e)}"