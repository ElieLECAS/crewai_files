import os
import io
import base64
import glob
import json
import re
import csv
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Annotated, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
from src.database import insert_document, is_document_exists

# Configuration du logging avec timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Chargement de l'environnement
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL = os.getenv("MODEL", "glm-ocr:q8_0")
MODEL_EXTRACT = os.getenv("MODEL_EXTRACT") or MODEL  # Modèle pour extraction (optionnel, défaut=MODEL)
TIMEOUT = int(os.getenv("TIMEOUT", "600"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))  # Nombre de fichiers à traiter en parallèle
# Limite de caractères du document envoyée à l'extraction (réduire si 500 avec glm-ocr)
EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "4000"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Chemins des volumes Docker
INPUT_DIR = "/app/input"

# Verrou pour sériaiser tous les appels Ollama (évite GGML crash avec glm-ocr en requêtes concurrentes)
OLLAMA_LOCK = threading.Lock()

# Fichiers en cours (partagé entre watcher et upload API pour éviter double traitement)
_files_in_progress_ref: Optional[set] = None

# Singleton ChatOllama pour éviter de recharger le modèle à chaque appel
_llm_instances: dict = {}


def get_llm(model: Optional[str] = None, format_json: bool = False) -> ChatOllama:
    """Retourne une instance ChatOllama réutilisable (singleton par modèle)."""
    global _llm_instances
    model = model or MODEL
    key = f"{model}:{format_json}"
    if key not in _llm_instances:
        kwargs = dict(
            model=model,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            temperature=0,
            num_ctx=OLLAMA_NUM_CTX,
        )
        if format_json:
            kwargs["format"] = "json"
        _llm_instances[key] = ChatOllama(**kwargs)
    return _llm_instances[key]


def is_local_small_model(model_name: str) -> bool:
    """Modèles vision 3B ou textuels 7B - prompts courts pour éviter surcharge contexte."""
    m = (model_name or "").lower()
    return "glm-ocr" in m or "deepseek-ocr" in m or ":7b" in m or ":3b" in m


def register_file_in_progress(file_path: str) -> None:
    """Marque un fichier comme en cours (upload API) pour que le watcher l'ignore."""
    if _files_in_progress_ref is not None:
        _files_in_progress_ref.add(os.path.abspath(file_path))


def unregister_file_in_progress(file_path: str) -> None:
    """Retire un fichier de la liste des en cours."""
    if _files_in_progress_ref is not None:
        _files_in_progress_ref.discard(os.path.abspath(file_path))


# ============================================================================
# SCHÉMAS PYDANTIC POUR L'EXTRACTION STRUCTURÉE
# ============================================================================

class LineItem(BaseModel):
    """Ligne d'article générique."""
    reference: Optional[str] = Field(None, description="Référence ou numéro de commande")
    designation: str = Field(..., description="Description complète de l'article")
    quantite: Optional[float] = Field(None, description="Quantité commandée")
    unite: Optional[str] = Field(None, description="Unité de mesure (Pièces, Mètres, etc.)")
    prix_unitaire_brut: Optional[float] = Field(None, description="Prix unitaire avant remise")
    remise_pourcentage: Optional[float] = Field(None, description="Pourcentage de remise appliqué")
    total_ht: Optional[float] = Field(None, description="Montant total HT de la ligne")
    tva_pourcentage: Optional[float] = Field(None, description="Taux de TVA applicable")


class InvoiceHeader(BaseModel):
    """En-tête de facture."""
    numero_facture: Optional[str] = Field(None, description="Numéro de facture")
    date: Optional[str] = Field(None, description="Date d'émission")
    client_nom: Optional[str] = Field(None, description="Nom du client")
    total_ttc: Optional[float] = Field(None, description="Montant total TTC")


# --- VITRAGLASS ---
class VitraglassLineItem(LineItem):
    """Ligne de facture spécifique à Vitraglass."""
    numero_commande: Optional[str] = Field(None, description="Numéro de commande Vitraglass")
    bon_livraison: Optional[str] = Field(None, description="Numéro du bon de livraison associé")
    reference_commande: Optional[str] = Field(None, description="Référence interne de la commande")
    hauteur_largeur: Optional[str] = Field(None, description="Dimensions Hauteur x Largeur")
    intercalaire: Optional[str] = Field(None, description="Type d'intercalaire (ex: 10TGNO)")
    surface: Optional[float] = Field(None, description="Surface unitaire")
    surface_totale: Optional[float] = Field(None, description="Surface totale pour la ligne")

class VitraglassInvoiceSchema(BaseModel):
    """Schéma spécifique pour les factures Vitraglass."""
    entete: InvoiceHeader = Field(..., description="En-tête de la facture")
    lignes: List[VitraglassLineItem] = Field(default_factory=list, description="Lignes de facturation détaillées")
    fichier_source: Optional[str] = Field(None, description="Nom du fichier source")


# --- PROFERM / SOPROFEN (format avec références SOI) ---
class ProfermLineItem(LineItem):
    """Ligne de facture spécifique à Proferm/Soprofen (format avec références SOI).
    
    Structure du tableau SOPROFEN (6 colonnes principales à extraire) :
    1. N° de commande (reference_soi) : Référence SOI complète (ex: "SOI C 25 212 004 993")
    2. Désignation (designation) : Description complète du produit (obligatoire)
    3. Quantité (quantite) : Nombre d'unités (ex: 1.0, 4.0)
    4. Unité (unite) : Unité de mesure (ex: "PIECE")
    5. P.U. Brut EUR (prix_unitaire_brut) : Prix unitaire avant remise (ex: 643.10)
    6. % Rem. (remise_pourcentage) : Pourcentage de remise appliqué (ex: 44.5)
    7. Total EUR (total_ht) : Montant total HT après remise (ex: 356.92)
    
    Tous les champs de base héritent de LineItem.
    """
    reference_soi: Optional[str] = Field(None, description="Référence SOI complète (N° de commande) - Colonne 1 du tableau SOPROFEN")
    dimensions: Optional[str] = Field(None, description="Dimensions extraites de la désignation (format: L x H mm ou L * H mm)")

class ProfermInvoiceSchema(BaseModel):
    """Schéma spécifique pour les factures Proferm/Soprofen."""
    entete: InvoiceHeader = Field(..., description="En-tête de la facture")
    lignes: List[ProfermLineItem] = Field(default_factory=list, description="Lignes de facturation")
    fichier_source: Optional[str] = Field(None, description="Nom du fichier source")


class InvoiceSchema(BaseModel):
    """Schéma complet pour une facture générique."""
    entete: InvoiceHeader = Field(..., description="Informations d'en-tête de la facture")
    lignes: List[LineItem] = Field(default_factory=list, description="Liste des lignes de facturation")
    fichier_source: Optional[str] = Field(None, description="Nom du fichier PDF source")


class QuoteHeader(BaseModel):
    """En-tête de devis."""
    numero_devis: Optional[str] = Field(None, description="Numéro du devis")
    date_emission: Optional[str] = Field(None, description="Date d'émission du devis")
    date_validite: Optional[str] = Field(None, description="Date de validité du devis")
    entreprise_nom: Optional[str] = Field(None, description="Nom de l'entreprise émettrice")
    client_nom: Optional[str] = Field(None, description="Nom du client destinataire")


class QuoteLineItem(LineItem):
    """Ligne de prestation spécifique pour devis."""
    dimensions: Optional[str] = Field(None, description="Dimensions (ex: 1400 x 2150 mm)")
    couleur_exterieure: Optional[str] = Field(None, description="Couleur extérieure (ex: Gris 7035)")
    couleur_interieure: Optional[str] = Field(None, description="Couleur intérieure")
    caracteristiques_techniques: List[str] = Field(default_factory=list, description="Liste des caractéristiques techniques (ex: Seuil 18mm, Serrure...)")


class QuoteTotals(BaseModel):
    """Totaux financiers d'un devis."""
    total_ht: Optional[float] = Field(None, description="Total Hors Taxes")
    total_tva: Optional[float] = Field(None, description="Montant total de la TVA")
    total_ttc: Optional[float] = Field(None, description="Total Toutes Taxes Comprises")


class QuoteSchema(BaseModel):
    """Schéma complet pour un devis."""
    entete: QuoteHeader = Field(..., description="Informations d'en-tête du devis")
    prestations: List[QuoteLineItem] = Field(default_factory=list, description="Liste des prestations ou services")
    totaux: QuoteTotals = Field(..., description="Totaux financiers du devis")
    conditions_paiement: Optional[str] = Field(None, description="Conditions et délais de paiement")
    fichier_source: Optional[str] = Field(None, description="Nom du fichier PDF source")


# ============================================================================
# ÉTAT DU GRAPHE (AgentState)
# ============================================================================

class AgentState(TypedDict):
    """État partagé entre les nœuds du graphe LangGraph."""
    file_path: str
    file_name: str
    doc_markdown: Optional[str]
    doc_type: Optional[str]  # "devis", "facture" ou "bon_livraison"
    supplier: Optional[str]   # "vitraglass", "soprofen", etc.
    table_structure: Optional[dict]  # {nb_colonnes: int, has_remise: bool, format_tableau: str}
    structured_data: Optional[dict]
    is_valid: bool
    retry_count: int
    error_message: Optional[str]


# ============================================================================
# EXEMPLES FEW-SHOT POUR STABILISER LE MODÈLE
# ============================================================================

INVOICE_EXAMPLE_VITRAGLASS = """{
  "entete": {
    "numero_facture": "2025 - 37655",
    "date": "31-10-2025",
    "client_nom": "PROFERM ALU",
    "total_ttc": 1206.50
  },
  "lignes": [
    {
      "numero_commande": "2025 044432",
      "bon_livraison": "2025 67804",
      "reference_commande": "2503133.NJ0",
      "designation": "D.V. : F 44/2 clair + Low-e 1.0 Advanced 6 mm(#3) +Gaz Argon",
      "hauteur_largeur": "2065 x 1727",
      "intercalaire": "10TGNO",
      "surface": 3.58,
      "surface_totale": 3.58,
      "quantite": 1.0,
      "prix_unitaire_brut": 141.33,
      "total_ht": 505.96,
      "tva_pourcentage": 20.0
    },
    {
      "numero_commande": "2025 044577",
      "bon_livraison": "2025 67804",
      "reference_commande": "2509551.NS0",
      "designation": "D.V. : Glace claire 6 mm + Low-e 4 mm(#3) +Gaz Argon + EMBALLAGE (MOUSSE + FILM)",
      "hauteur_largeur": "2018 x 929",
      "intercalaire": "14TGNO",
      "surface": 1.88,
      "surface_totale": 1.88,
      "quantite": 1.0,
      "prix_unitaire_brut": 47.40,
      "total_ht": 89.11,
      "tva_pourcentage": 20.0
    },
    {
      "numero_commande": "2025 582010",
      "bon_livraison": "2025 67804",
      "reference_commande": "49181/CDE 2510513.Y01/00175 SECHER D19777/POS201",
      "designation": "D.V. : Glace claire 6 mm + Low-e 4 mm(#3) +Gaz Argon",
      "hauteur_largeur": "1922 x 939",
      "intercalaire": "14TGNO",
      "surface": 0.87,
      "surface_totale": 0.87,
      "quantite": 1.0,
      "prix_unitaire_brut": 29.84,
      "total_ht": 25.96,
      "tva_pourcentage": 20.0
    }
  ],
  "fichier_source": "FACPDF_2025_37655.pdf"
}"""

INVOICE_EXAMPLE_PROFERM = """{
  "entete": {
    "numero_facture": "W0848335",
    "date": "2026-01-12",
    "client_nom": "PROFERM MULTITECHNIQUES",
    "total_ttc": 1143.5
  },
  "lignes": [
    {
      "reference_soi": "SOI C 25 212 001 555",
      "designation": "Ligne 1 Coffre Paco Dimension Tableau (L x H mm) : 1390 * 2100 Blanc R=0.18",
      "dimensions": "1390 * 2100 mm",
      "quantite": 1.0,
      "unite": "PIECE",
      "prix_unitaire_brut": 1183.11,
      "remise_pourcentage": 64.0,
      "total_ht": 425.92,
      "tva_pourcentage": 20.0
    },
    {
      "reference_soi": "SOI C 25 212 001 555",
      "designation": "Ligne 2 Coffre Paco Dimension Tableau (L x H mm) : 790 * 970 Blanc R=0.18",
      "dimensions": "790 * 970 mm",
      "quantite": 1.0,
      "unite": "PIECE",
      "prix_unitaire_brut": 919.11,
      "remise_pourcentage": 64.0,
      "total_ht": 330.88,
      "tva_pourcentage": 20.0
    },
    {
      "reference_soi": "SOI C 25 210 000 406",
      "designation": "Ligne 1 Acc à la commande MX10-55 Adaptateur 55mm, 2984mm par 1 pièces",
      "quantite": 2.984,
      "unite": "Mètres",
      "prix_unitaire_brut": 19.91,
      "remise_pourcentage": 45.0,
      "total_ht": 32.68,
      "tva_pourcentage": 20.0
    }
  ],
  "fichier_source": "W0848335_Demat.pdf"
}"""

QUOTE_EXAMPLE_OPTIMIZED = """{
  "entete": {
    "numero_devis": "D47025",
    "date_emission": "2025-10-24",
    "date_validite": "2025-11-07",
    "entreprise_nom": "PROFERM MULTITECHNIQUES",
    "client_nom": "Société LE LOFT"
  },
  "prestations": [
    {
      "designation": "Porte d'entrée vitrée 2 vantaux tiercés",
      "dimensions": "Larg 1400 mm x Haut 2150 mm",
      "couleur_exterieure": "Gris 7035 Granité",
      "couleur_interieure": "Gris 7035 Granité",
      "caracteristiques_techniques": [
        "Pose en applique intérieure",
        "Dormant ep.55mm",
        "Seuil 18 mm",
        "Vitrage sécurité 44.2/12/4 faible émissif argon",
        "Ferme-porte"
      ],
      "quantite": 1.0,
      "unite": "Unité",
      "prix_unitaire_brut": 3583.6,
      "total_ht": 3583.6,
      "tva_pourcentage": 20.0
    }
  ],
  "totaux": {
    "total_ht": 7915.62,
    "total_tva": 1583.12,
    "total_ttc": 9498.74
  },
  "conditions_paiement": "selon ouverture de compte",
  "fichier_source": "Devis D47025 - PE.pdf"
}"""

# Exemples courts pour petits modèles (deepseek-ocr, glm-ocr)
INVOICE_EXAMPLE_VITRAGLASS_SHORT = """{"entete":{"numero_facture":"2025-37655","date":"31-10-2025","client_nom":"PROFERM ALU","total_ttc":1206.50},"lignes":[{"numero_commande":"2025 044432","bon_livraison":"2025 67804","reference_commande":"2503133.NJ0","designation":"D.V. : F 44/2 clair + Low-e 6 mm","hauteur_largeur":"2065 x 1727","intercalaire":"10TGNO","surface":3.58,"surface_totale":3.58,"quantite":1.0,"prix_unitaire_brut":141.33,"total_ht":505.96,"tva_pourcentage":20.0}],"fichier_source":"FACPDF.pdf"}"""

INVOICE_EXAMPLE_PROFERM_SHORT = """{"entete":{"numero_facture":"W0848335","date":"2026-01-12","client_nom":"PROFERM","total_ttc":1143.5},"lignes":[{"reference_soi":"SOI C 25 212 001 555","designation":"Coffre Paco 1390*2100 Blanc","dimensions":"1390*2100 mm","quantite":1.0,"unite":"PIECE","prix_unitaire_brut":1183.11,"remise_pourcentage":64.0,"total_ht":425.92,"tva_pourcentage":20.0}],"fichier_source":"W0848335.pdf"}"""

QUOTE_EXAMPLE_SHORT = """{"entete":{"numero_devis":"D47025","date_emission":"2025-10-24","client_nom":"LE LOFT"},"prestations":[{"designation":"Porte vitrée 2 vantaux","dimensions":"1400x2150 mm","quantite":1.0,"prix_unitaire_brut":3583.6,"total_ht":3583.6,"tva_pourcentage":20.0}],"totaux":{"total_ht":7915.62,"total_ttc":9498.74},"fichier_source":"Devis.pdf"}"""


# ============================================================================
# OUTILS PDF (EXTRACTEUR DE TEXTE + OCR)
# ============================================================================

def extract_text_native(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extrait le texte natif du PDF page par page.
    Retourne une liste de tuples (numéro_page, texte).
    """
    pages_content = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_content.append((i + 1, text))
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction native du PDF : {str(e)}")
    return pages_content

def is_page_text_empty(text: str, threshold: int = 20) -> bool:
    """
    Détermine si une page est considérée comme vide ou scannée
    basé sur le nombre de caractères extraits.
    """
    clean_text = text.strip()
    return len(clean_text) < threshold

def ocr_page(pdf_path: str, page_num: int) -> str:
    """
    Convertit une page du PDF en image et effectue l'OCR via glm-ocr (vision LLM).
    """
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            fmt="jpeg"
        )
        if not images:
            return ""
        pil_image = images[0]
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"
        llm = get_llm(model=MODEL)
        # deepseek-ocr attend des prompts courts et précis (doc Ollama)
        if "deepseek-ocr" in (MODEL or "").lower():
            prompt = "\nExtract the text in the image."
        else:
            prompt = "Extrais tout le texte de cette page de document, dans l'ordre. Retourne uniquement le texte brut, sans markdown."
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
        with OLLAMA_LOCK:
            response = llm.invoke([msg])
        return (response.content or "").strip()
    except Exception as e:
        logger.error(f"Erreur lors de l'OCR de la page {page_num} : {str(e)}")
        return ""

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def clean_ocr_text(text: str) -> str:
    """
    Nettoie le texte OCR pour supprimer les caractères bruités et les séquences répétées.
    Supprime les séquences de caractères spéciaux multiples (ex: |||||, ====) et les caractères non-ASCII problématiques.
    """
    if not text:
        return text
    
    # Supprimer les séquences répétées de caractères spéciaux (3+ répétitions)
    # Exemples: |||||, ====, ----, ~~~~, etc.
    text = re.sub(r'([|=\-~_\.\+\*#]{3,})', '', text)
    
    # Supprimer les séquences répétées d'autres caractères spéciaux courants en OCR
    text = re.sub(r'([^\w\s]{3,})', '', text)  # 3+ caractères non-alphanumériques consécutifs
    
    # Supprimer les caractères de contrôle (sauf les sauts de ligne et tabulations)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Supprimer les espaces multiples (garder max 2 espaces consécutifs)
    text = re.sub(r' {3,}', '  ', text)
    
    # Supprimer les lignes vides multiples (garder max 2 lignes vides consécutives)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Nettoyer les caractères Unicode problématiques courants en OCR
    # Remplacer certains caractères Unicode similaires par leurs équivalents ASCII
    replacements = {
        '\u2018': "'",  # ' (apostrophe courbe gauche)
        '\u2019': "'",  # ' (apostrophe courbe droite)
        '\u201C': '"',  # " (guillemet courbe gauche)
        '\u201D': '"',  # " (guillemet courbe droit)
        '\u2013': '-',  # – (tiret en)
        '\u2014': '-',  # — (tiret em)
        '\u2026': '...',  # … (points de suspension)
        '\u00A0': ' ',  # (espace insécable)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()


def _extract_json_from_text(text: str) -> dict:
    """Extrait un objet JSON du texte (bloc ```json ... ``` ou premier { ... })."""
    if not text or not text.strip():
        raise ValueError("Réponse vide")
    text = text.strip()
    # Bloc markdown ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return json.loads(match.group(1).strip())
    # Premier objet JSON
    start = text.find("{")
    if start == -1:
        raise ValueError("Aucun JSON trouvé dans la réponse")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("JSON mal formé dans la réponse")


# ============================================================================
# FONCTIONS UTILITAIRES DE NORMALISATION
# ============================================================================

def normalize_vitraglass_line(line: dict) -> List[dict]:
    """
    Normalise une ligne Vitraglass : si des champs sont des listes, crée plusieurs lignes.
    Retourne une liste de lignes normalisées avec une ligne par pièce.
    """
    # Vérifier si des champs critiques sont des listes
    list_fields = ['hauteur_largeur', 'intercalaire', 'surface', 'prix_unitaire_brut']
    has_lists = any(isinstance(line.get(field), list) for field in list_fields)
    
    if not has_lists:
        return [line]
    
    # Trouver la longueur maximale des listes (nombre de pièces différentes)
    max_len = 1
    for field in list_fields:
        value = line.get(field)
        if isinstance(value, list):
            max_len = max(max_len, len(value))
    
    # Créer une ligne séparée par pièce
    normalized_lines = []
    original_qty = line.get('quantite', 1.0)
    original_total_ht = line.get('total_ht', 0.0)
    
    for i in range(max_len):
        new_line = line.copy()
        
        # Extraire les valeurs des listes, ou utiliser la valeur simple
        for field in list_fields:
            value = line.get(field)
            if isinstance(value, list):
                new_line[field] = value[i] if i < len(value) else (value[0] if value else None)
            else:
                new_line[field] = value
        
        # Ajuster quantité : 1 pièce par ligne
        new_line['quantite'] = 1.0
        
        # Recalculer total_ht basé sur prix_unitaire_brut et quantité
        if new_line.get('prix_unitaire_brut') and isinstance(new_line['prix_unitaire_brut'], (int, float)):
            new_line['total_ht'] = new_line['prix_unitaire_brut'] * new_line['quantite']
        else:
            # Si on ne peut pas calculer, diviser le total HT par le nombre de pièces
            if isinstance(original_total_ht, (int, float)) and original_total_ht > 0:
                new_line['total_ht'] = original_total_ht / max_len
        
        # Ajuster surface_totale si nécessaire
        if new_line.get('surface') and isinstance(new_line['surface'], (int, float)):
            new_line['surface_totale'] = new_line['surface'] * new_line['quantite']
        
        normalized_lines.append(new_line)
    
    return normalized_lines if normalized_lines else [line]


def restructure_json_data(raw_data: dict, supplier: str, doc_type: str) -> dict:
    """
    Restructure le JSON si le modèle a retourné une structure plate au lieu de entete/lignes.
    """
    if not raw_data or not isinstance(raw_data, dict):
        return raw_data
    
    # Si déjà bien structuré, retourner tel quel
    if "entete" in raw_data and ("lignes" in raw_data or "prestations" in raw_data):
        return raw_data
    
    restructured = {"fichier_source": raw_data.get("fichier_source")}
    
    # Champs d'en-tête (facture)
    entete_fields = ["numero_facture", "date", "client_nom", "total_ttc"]
    # Champs d'en-tête (devis)
    entete_devis_fields = ["numero_devis", "date_emission", "date_validite", "entreprise_nom", "client_nom"]
    
    if doc_type == "facture":
        entete = {}
        for field in entete_fields:
            if field in raw_data:
                entete[field] = raw_data.pop(field)
        restructured["entete"] = entete if entete else {"numero_facture": None, "date": None, "client_nom": None, "total_ttc": None}
        
        # Détecter les lignes : si on a des champs de ligne au niveau racine ou une liste
        lignes = []
        if "lignes" in raw_data and isinstance(raw_data["lignes"], list):
            lignes = raw_data["lignes"]
        elif any(field in raw_data for field in ["reference_soi", "designation", "quantite", "prix_unitaire_brut"]):
            # Créer une ligne depuis les champs racine
            ligne = {}
            ligne_fields = ["reference_soi", "designation", "quantite", "unite", "prix_unitaire_brut", 
                          "remise_pourcentage", "total_ht", "tva_pourcentage", "dimensions"]
            for field in ligne_fields:
                if field in raw_data:
                    ligne[field] = raw_data.pop(field)
            if ligne:
                lignes = [ligne]
        restructured["lignes"] = lignes
    
    elif doc_type == "devis":
        entete = {}
        for field in entete_devis_fields:
            if field in raw_data:
                entete[field] = raw_data.pop(field)
        restructured["entete"] = entete if entete else {}
        
        if "prestations" in raw_data and isinstance(raw_data["prestations"], list):
            restructured["prestations"] = raw_data["prestations"]
        else:
            restructured["prestations"] = []
        
        if "totaux" in raw_data:
            restructured["totaux"] = raw_data["totaux"]
        else:
            restructured["totaux"] = {}
    
    return restructured


def normalize_extracted_data(structured_data: dict, supplier: str) -> dict:
    """
    Normalise les données extraites pour corriger les erreurs de format (listes au lieu de valeurs simples).
    """
    if supplier != "vitraglass" or not structured_data:
        return structured_data
    
    lines = structured_data.get("lignes", [])
    if not lines:
        return structured_data
    
    normalized_lines = []
    for line in lines:
        normalized = normalize_vitraglass_line(line)
        normalized_lines.extend(normalized)
    
    structured_data["lignes"] = normalized_lines
    return structured_data


# ============================================================================
# NŒUDS DU GRAPHE LANGGRAPH
# ============================================================================

def partition_node(state: AgentState) -> AgentState:
    """
    Nœud 1 : Partitionne le PDF de manière hybride.
    1. Extrait le texte natif.
    2. Si une page est vide/scannée, utilise l'OCR via glm-ocr (vision LLM).
    """
    try:
        file_path = state["file_path"]
        file_name = state["file_name"]
        
        logger.info(f"📄 Analyse hybride du document : {file_name}")
        start_time = datetime.now()
        
        # 1. Extraction native
        native_pages = extract_text_native(file_path)
        full_content = []
        
        for page_num, text in native_pages:
            if is_page_text_empty(text):
                logger.info(f"   Page {page_num} : Pas de texte détecté, passage à l'OCR...")
                ocr_text = ocr_page(file_path, page_num)
                # Nettoyer le texte OCR avant de l'ajouter
                ocr_text = clean_ocr_text(ocr_text)
                full_content.append(f"## Page {page_num} (OCR)\n\n{ocr_text}")
            else:
                logger.info(f"   Page {page_num} : Texte natif extrait")
                # Nettoyer aussi le texte natif (peut contenir des artefacts)
                text = clean_ocr_text(text)
                full_content.append(f"## Page {page_num}\n\n{text}")
        
        doc_markdown = "\n\n".join(full_content)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Document analysé ({len(doc_markdown)} caractères) en {elapsed:.2f}s")
        
        if not doc_markdown.strip():
            raise Exception("Le document semble vide après analyse native et OCR")
        
        return {
            **state,
            "doc_markdown": doc_markdown,
            "error_message": None
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse hybride : {str(e)}")
        return {
            **state,
            "doc_markdown": None,
            "error_message": f"Analyse échouée : {str(e)}"
        }


def router_supplier_node(state: AgentState) -> AgentState:
    """
    Nœud 2 : Détecte type (devis/facture/BL) et fournisseur en un seul appel LLM.
    Pas de règles ni fallback — tout par l'IA.
    """
    try:
        doc_markdown = state["doc_markdown"]
        file_name = state["file_name"]
        
        if not doc_markdown:
            logger.warning("Pas de contenu à analyser")
            return {**state, "doc_type": "devis", "supplier": "inconnu", "table_structure": {"nb_colonnes": 0, "has_remise": False, "format_tableau": ""}}
        
        logger.info(f"🔍 Détection type + fournisseur : {file_name}")
        
        ctx_chars = 2000 if is_local_small_model(MODEL) else 3000
        prompt = f"""Tu analyses un document. Identifie TYPE, FOURNISSEUR, structure du tableau.

RÈGLES FOURNISSEUR (facture uniquement) :
- L'émetteur = l'entreprise dans l'en-tête (nom, adresse, SIRET)
- VITRAGLASS, GROUPE DEVGLASS, CEKAL, GLASS A LIA = fournisseur vitraglass
- SOPROFEN = fournisseur soprofen si présent en en-tête avec adresse
- PROFERM, PROFERM ALU, PROFERM MULTITECHNIQUES = CLIENT (notre entreprise), PAS fournisseur
- Pour devis ou bon de livraison : fournisseur = inconnu, nb_col=0, has_remise=non

Pour une FACTURE : compte le nombre de colonnes du tableau principal (ex: SOPROFEN=7, VITRAGLASS=8) et indique si une colonne remise existe (oui/non).

CONTENU :
{doc_markdown[:ctx_chars]}

Réponds UNIQUEMENT au format : TYPE|FOURNISSEUR|NB_COL|HAS_REMISE
Exemples : facture|soprofen|7|oui   facture|vitraglass|8|non   devis|inconnu|0|non

RÉPONSE :"""
        
        llm = get_llm(model=MODEL)
        with OLLAMA_LOCK:
            logger.info("   Appel Ollama (type + fournisseur)…")
            response = llm.invoke([HumanMessage(content=prompt)])
        
        raw = (response.content or "").strip().lower()
        doc_type = "devis"
        supplier = "inconnu"
        table_structure = {"nb_colonnes": 0, "has_remise": False, "format_tableau": ""}
        
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 2:
            type_part = parts[0] or ""
            supp_part = parts[1] or ""
            if "facture" in type_part:
                doc_type = "facture"
            elif "bon" in type_part or "livraison" in type_part or "bl" in type_part:
                doc_type = "bon_livraison"
            elif "devis" in type_part:
                doc_type = "devis"
            if doc_type == "facture":
                if "vitraglass" in supp_part:
                    supplier = "vitraglass"
                elif "soprofen" in supp_part:
                    supplier = "soprofen"
            if len(parts) >= 4 and doc_type == "facture":
                try:
                    nb_col = int(parts[2]) if parts[2].isdigit() else 0
                    has_remise = "oui" in (parts[3] or "")
                    table_structure = {
                        "nb_colonnes": nb_col,
                        "has_remise": has_remise,
                        "format_tableau": f"{supplier}_{nb_col}col" if supplier != "inconnu" else ""
                    }
                except (ValueError, IndexError):
                    pass
        else:
            if "facture" in raw:
                doc_type = "facture"
            elif "devis" in raw:
                doc_type = "devis"
            elif "bon" in raw or "livraison" in raw:
                doc_type = "bon_livraison"
            if doc_type == "facture":
                if "vitraglass" in raw:
                    supplier = "vitraglass"
                elif "soprofen" in raw:
                    supplier = "soprofen"
        
        logger.info(f"✅ Type: {doc_type}, Fournisseur: {supplier}, Structure: {table_structure}")
        return {**state, "doc_type": doc_type, "supplier": supplier, "table_structure": table_structure}
    
    except Exception as e:
        logger.error(f"❌ Erreur détection : {str(e)}")
        return {**state, "doc_type": "devis", "supplier": "inconnu", "table_structure": {"nb_colonnes": 0, "has_remise": False, "format_tableau": ""}}


def reformat_markdown_node(state: AgentState) -> AgentState:
    """
    Nœud intermédiaire : Réécrit le texte OCR bruité en tableau Markdown propre.
    Exécuté uniquement pour factures SOPROFEN/VITRAGLASS.
    """
    doc_markdown = state.get("doc_markdown")
    doc_type = state.get("doc_type")
    supplier = state.get("supplier", "inconnu")
    
    if not doc_markdown or doc_type != "facture" or supplier not in ("soprofen", "vitraglass"):
        return state
    
    try:
        logger.info(f"📐 Reformattage Markdown (facture {supplier})...")
        prompt = f"""Le texte ci-dessous provient d'un document scanné. Réécris-le en tableau Markdown propre :
- Une ligne d'en-tête avec les noms de colonnes séparés par |
- Une ligne par ligne de données, colonnes alignées
- Garde tout le contenu, ne supprime rien
- Corrige les décalages et caractères parasites (|, _)

TEXTE :
{doc_markdown[:3000]}"""
        
        llm = get_llm(model=MODEL)
        with OLLAMA_LOCK:
            response = llm.invoke([HumanMessage(content=prompt)])
        
        reformed = (response.content or "").strip()
        if reformed and len(reformed) > 50:
            logger.info("✅ Markdown reformatté")
            return {**state, "doc_markdown": reformed}
    except Exception as e:
        logger.warning(f"⚠️ Reformattage Markdown échoué : {e}, conservation du texte original")
    return state


def extract_node(state: AgentState) -> AgentState:
    """
    Nœud 3 : Extrait les données structurées via Pydantic et Few-Shot.
    Sélectionne le schéma optimisé selon le type et le fournisseur.
    """
    try:
        doc_markdown = state["doc_markdown"]
        doc_type = state["doc_type"]
        supplier = state.get("supplier", "inconnu")
        table_structure = state.get("table_structure") or {}
        file_name = state["file_name"]
        
        structure_hint = ""
        if table_structure.get("nb_colonnes") and doc_type == "facture":
            structure_hint = f"\nStructure détectée : {table_structure.get('nb_colonnes', 0)} colonnes, remise : {'oui' if table_structure.get('has_remise') else 'non'}\n"
        
        logger.info(f"🤖 Extraction des données structurées (type: {doc_type}, fournisseur: {supplier})")
        logger.info(f"   ⏳ Extraction en cours (modèle {MODEL_EXTRACT}) — peut prendre 1 à 3 min sur CPU…")
        
        if not doc_markdown:
            logger.error("❌ Pas de contenu Markdown disponible pour l'extraction")
            return {
                **state,
                "structured_data": None,
                "error_message": "Analyse échouée : pas de contenu Markdown disponible",
                "retry_count": state.get("retry_count", 0) + 1
            }
        
        # Créer le LLM avec structured output
        llm = get_llm(model=MODEL_EXTRACT, format_json=True)
        
        # Contexte réduit pour modèles locaux (7B/3B, vision)
        if is_local_small_model(MODEL_EXTRACT):
            extract_chars = min(EXTRACT_MAX_CHARS, 2500)
        else:
            extract_chars = EXTRACT_MAX_CHARS
        
        # Sélection du schéma, exemple et System Prompt selon doc_type et supplier
        use_short = is_local_small_model(MODEL_EXTRACT)
        system_prompt = ""
        if doc_type == "facture":
            if supplier == "vitraglass":
                schema_class = VitraglassInvoiceSchema
                example = INVOICE_EXAMPLE_VITRAGLASS_SHORT if use_short else INVOICE_EXAMPLE_VITRAGLASS
                system_prompt = """Tu es un extracteur de factures VITRAGLASS. Les dimensions "Hauteur x Largeur" sont AU MILIEU de la désignation (format "2065 x 1727").
Extrais-les dans hauteur_largeur. La désignation complète va dans designation (D.V. : ...)."""
                instr_supp = """- Num→reference_commande, Hauteur x Largeur→hauteur_largeur, Intercalaire→intercalaire, Surface→surface, Prix Unitaire→prix_unitaire_brut, Montant→total_ht. Convertis "3,58"→3.58.""" if use_short else """- CRITIQUE : Le tableau VITRAGLASS a ces colonnes dans cet ordre :
  COLONNE 1 : "Num" → reference ou reference_commande (numéro de ligne : 001, 002, 003...)
  COLONNE 2 : "Qté" → quantite (toujours 1 pour les vitres individuelles)
  COLONNE 3 : "Hauteur x Largeur" → hauteur_largeur (format: "2065 x 1727" ou "1922 x 939")
  COLONNE 4 : "Intercalaire" → intercalaire (ex: "10TGNO", "14TGNO", "16TGNO")
  COLONNE 5 : "Surface" → surface (surface unitaire en m², ex: 3.58, 0.87, 1.88)
  COLONNE 6 : "Surface Totale" → surface_totale (surface totale, souvent = surface si qté=1)
  COLONNE 7 : "Prix Unitaire" → prix_unitaire_brut (prix au m², ex: 141.33, 47.40, 29.84)
  COLONNE 8 : "Montant" → total_ht (montant total de la ligne, ex: 505.96, 89.11, 25.96)

INFORMATIONS COMPLÉMENTAIRES À EXTRAIRE :
- numero_commande : cherche "COMMANDE NUMERO" suivi du numéro (ex: "2025 044432", "2025 582010")
- bon_livraison : cherche "BON DE LIVRAISON :" suivi du numéro (ex: "2025 67804")
- reference_commande : cherche "Référence :" ou "Référence cmde :" suivi de la référence
- designation : cherche "D.V. :" suivi de la description du vitrage (ex: "D.V. : F 44/2 clair + Low-e 1.0 Advanced 6 mm(#3) +Gaz Argon")

RÈGLES IMPORTANTES :
- Chaque ligne avec un numéro (001, 002, 003...) dans la colonne "Num" est UNE LIGNE SÉPARÉE
- Si plusieurs lignes ont la même commande, c'est normal (même numero_commande, bon_livraison)
- IGNORE les lignes "Pièce :", "Sous total :", "Sous-total H.T. commande"
- Convertis les nombres : "3,58" → 3.58, "141,33" → 141.33"""
            elif supplier == "soprofen":
                schema_class = ProfermInvoiceSchema
                example = INVOICE_EXAMPLE_PROFERM_SHORT if use_short else INVOICE_EXAMPLE_PROFERM
                system_prompt = """Tu es un extracteur de factures SOPROFEN.

ZÉRO CALCUL : NE JAMAIS calculer, recalculer ou déduire. Tu COPIES UNIQUEMENT les valeurs telles qu'elles apparaissent dans chaque colonne du document. Aucune formule, aucun calcul.

ORDRE DES COLONNES (position = numéro de colonne dans le tableau) :
1. N° de commande → reference_soi
2. Désignation → designation
3. Qte → quantite (nombre)
4. Unite → unite (texte)
5. P.U. Brut EUR → prix_unitaire_brut (copier la valeur de CETTE colonne, ex: 565,32 → 565.32)
6. % Rem. → remise_pourcentage
7. Total EUR → total_ht (copier la valeur de CETTE colonne, ex: 313,75 → 313.75)

RÈGLE : Chaque valeur va dans son champ. La colonne 5 = prix_unitaire_brut. La colonne 7 = total_ht. Ne pas inverser.
Exemple concret : si le document affiche P.U. Brut = 565,32 et Total EUR = 313,75 → prix_unitaire_brut=565.32, total_ht=313.75 (copier, ne pas calculer).

CRITIQUE - NE PAS MÉLANGER :
- Extrais UNIQUEMENT les lignes qui ont un "N° de commande" commençant par "SOI" (ex: SOI C 25 212 004 653).
- IGNORE la ligne "Sous totaux :" — ses valeurs (687,36, 351,34...) sont des SOMMES. Ne les mets JAMAIS dans les lignes d'articles.
- % Rem. = TOUJOURS un pourcentage entre 0 et 100 (ex: 44.5, 60, 64). Si tu vois 273 ou 351, c'est une ERREUR (ce sont des montants EUR, pas des %).
- Chaque ligne d'article a ses propres valeurs : P.U. Brut, % Rem., Total EUR. Ne pas copier les sous-totaux dans les lignes.

IGNORER : "Dont éco-contribution PMCB", "Sous totaux :"."""
                instr_supp = """- ZÉRO CALCUL : copie les valeurs. Colonne 5→prix_unitaire_brut, colonne 7→total_ht. Ne pas inverser ni calculer.""" if use_short else """- ZÉRO CALCUL : copie les valeurs telles quelles. AUCUN calcul. CRITIQUE : Le tableau SOPROFEN a EXACTEMENT ces colonnes dans cet ordre :
  COLONNE 1 : "N° de commande" → reference_soi (ex: "SOI C 25 212 001 555")
  COLONNE 2 : "Désignation" → designation (ex: "Ligne 1 Coffre Paco Dimension Tableau (L x H mm) : 1390 * 2100 Blanc R=0.18")
  COLONNE 3 : "Qte" → quantite (NOMBRE, ex: 1,000 → 1.0 ou 2,984 → 2.984)
  COLONNE 4 : "Unite" → unite (TEXTE, ex: "PIECE" ou "Mètres")
  COLONNE 5 : "P.U. Brut EUR" → prix_unitaire_brut (PRIX AVANT REMISE, ex: 1183,11 → 1183.11 ou 19,91 → 19.91)
  COLONNE 6 : "% Rem." → remise_pourcentage (POURCENTAGE, ex: 64 → 64.0 ou 45 → 45.0)
  COLONNE 7 : "Total EUR" → total_ht (PRIX FINAL APRÈS REMISE, ex: 425,92 → 425.92 ou 32,68 → 32.68)

RÈGLES ABSOLUES - AUCUN CALCUL :
- ZÉRO CALCUL : copie les valeurs telles quelles. Ne calcule jamais total_ht à partir de prix_unitaire_brut et remise.
- Colonne 5 "P.U. Brut EUR" → prix_unitaire_brut : copier le nombre de cette colonne (ex: 565,32 → 565.32)
- Colonne 7 "Total EUR" → total_ht : copier le nombre de cette colonne (ex: 313,75 → 313.75)
- NE PAS inverser : si tu vois 565,32 dans la colonne P.U. Brut et 313,75 dans Total EUR, garde-les ainsi.
- La colonne "Qte" (1,000) = quantité, pas un prix.

CRITIQUE - NE PAS MÉLANGER LIGNES ET SOUS-TOTAUX :
- Extrais UNIQUEMENT les lignes avec "N° de commande" = SOI C ... (ex: SOI C 25 212 004 653).
- IGNORE la ligne "Sous totaux :" — 687,36 et 351,34 sont des SOMMES. Ne les mets JAMAIS dans prix_unitaire_brut ou total_ht d'une ligne d'article.
- remise_pourcentage = TOUJOURS entre 0 et 100 (44.5, 60, 64). Jamais 273 ou 351 (ce sont des montants EUR).
- Chaque ligne = ses propres valeurs. Ligne 1 : 492,86 / 44.5 / 273,54. Ligne 2 : 194,50 / 60 / 77,80.

IGNORE : "Dont éco-contribution PMCB", "Sous totaux :", lignes vides.

Extrais les dimensions depuis la désignation (format "L x H mm" ou "L * H mm")."""
            else:
                # Schéma par défaut pour fournisseurs inconnus
                schema_class = ProfermInvoiceSchema
                example = INVOICE_EXAMPLE_PROFERM_SHORT if use_short else INVOICE_EXAMPLE_PROFERM
                instr_supp = "- Schéma standard facture." if use_short else "- Utilise le schéma standard d'extraction de facture."

            if use_short:
                prompt = f"""EXEMPLE (STRUCTURE OBLIGATOIRE : {{"entete": {{...}}, "lignes": [...]}}):
{example}
{structure_hint}
CONTENU :
{doc_markdown[:extract_chars]}

RÈGLES : {instr_supp}
STRUCTURE JSON REQUISE : {{"entete": {{"numero_facture": "...", "date": "...", "client_nom": "...", "total_ttc": ...}}, "lignes": [{{"reference_soi": "...", "designation": "...", ...}}]}}
Fichier : {file_name}
Retourne JSON valide uniquement."""
            else:
                prompt = f"""Tu es un expert en extraction de factures PDF {supplier.upper()}.
{structure_hint}
IMPORTANT : La structure JSON DOIT être {{"entete": {{...}}, "lignes": [...]}} - JAMAIS de champs au niveau racine.

EXEMPLE DE SORTIE ATTENDUE (STRUCTURE OBLIGATOIRE) :
{example}

ANALYSE DE L'EXEMPLE (AUCUN CALCUL - VALEURS COPIÉES DU DOCUMENT) :
- reference_soi = colonne 1 "N° de commande"
- designation = colonne 2 "Désignation"
- quantite = colonne 3 "Qte" (1,000 → 1.0)
- unite = colonne 4 "Unite" ("PIECE")
- prix_unitaire_brut = colonne 5 "P.U. Brut EUR" (1183,11 → 1183.11) - COPIER TEL QUEL
- remise_pourcentage = colonne 6 "% Rem." (64 → 64.0)
- total_ht = colonne 7 "Total EUR" (425,92 → 425.92) - COPIER TEL QUEL, NE PAS CALCULER

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:extract_chars]}

INSTRUCTIONS STRICTES :
{instr_supp}

CONVERSION DES NOMBRES (TRÈS IMPORTANT) :
- "1,000" → 1.0 (quantité)
- "2,984" → 2.984 (quantité)
- "1 183,11" ou "1183,11" → 1183.11 (prix unitaire brut)
- "919,11" → 919.11 (prix unitaire brut)
- "425,92" → 425.92 (total EUR)
- "330,88" → 330.88 (total EUR)
- "64" → 64.0 (remise %)
- "45" → 45.0 (remise %)

VÉRIFICATION : as-tu COPIÉ les valeurs sans calcul ? (prix_unitaire_brut = colonne 5, total_ht = colonne 7)

Le fichier source est : {file_name}

STRUCTURE JSON OBLIGATOIRE :
{{"entete": {{"numero_facture": "...", "date": "...", "client_nom": "...", "total_ttc": ...}}, "lignes": [{{"reference_soi": "...", "designation": "...", "quantite": ..., "unite": "...", "prix_unitaire_brut": ..., "remise_pourcentage": ..., "total_ht": ..., "tva_pourcentage": ...}}]}}

Retourne UNIQUEMENT le JSON valide avec cette structure exacte, sans commentaires."""

        else:  # devis
            schema_class = QuoteSchema
            example = QUOTE_EXAMPLE_SHORT if use_short else QUOTE_EXAMPLE_OPTIMIZED
            if use_short:
                prompt = f"""EXEMPLE :
{example}

CONTENU :
{doc_markdown[:extract_chars]}

Extrais prestations, totaux. Fichier : {file_name}
Retourne JSON valide uniquement."""
            else:
                prompt = f"""Tu es un expert en extraction de devis PDF PROFERM.

EXEMPLE DE SORTIE ATTENDUE :
{example}

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:extract_chars]}

INSTRUCTIONS :
- Extrais toutes les prestations/services du devis
- Détaille les dimensions, couleurs (ext/int) et caractéristiques techniques (sous forme de liste)
- Calcule les totaux HT, TVA et TTC
- Le fichier source est : {file_name}

STRUCTURE JSON OBLIGATOIRE : {{"entete": {{...}}, "prestations": [...], "totaux": {{...}}, "fichier_source": "..."}}
Retourne UNIQUEMENT le JSON valide avec cette structure exacte, sans commentaires."""
        
        # Messages pour l'appel LLM (SystemMessage si expertise fournisseur)
        extract_messages = [HumanMessage(content=prompt)]
        if system_prompt:
            extract_messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        
        # Extraction avec format=json (grammar strict) pour garantir JSON valide même sous charge CPU
        if is_local_small_model(MODEL_EXTRACT):
            with OLLAMA_LOCK:
                try:
                    raw_llm = get_llm(model=MODEL_EXTRACT, format_json=True)
                    result_raw = raw_llm.invoke(extract_messages)
                    raw_data = _extract_json_from_text(result_raw.content or "")
                    
                    # Restructurer si nécessaire (le modèle peut retourner une structure plate)
                    raw_data = restructure_json_data(raw_data, supplier, doc_type)
                    raw_data["fichier_source"] = file_name
                    raw_data = normalize_extracted_data(raw_data, supplier)
                    
                    if supplier == "vitraglass":
                        validated = VitraglassInvoiceSchema(**raw_data)
                        structured_data = validated.model_dump()
                    elif supplier == "soprofen":
                        validated = ProfermInvoiceSchema(**raw_data)
                        structured_data = validated.model_dump()
                    else:
                        structured_data = raw_data
                    logger.info("✅ Extraction terminée (format=json)")
                    return {
                        **state,
                        "structured_data": structured_data,
                        "error_message": None
                    }
                except Exception as e:
                    logger.error(f"❌ Extraction petit modèle échouée : {str(e)}")
                    logger.debug(f"   JSON brut reçu : {result_raw.content[:500] if 'result_raw' in locals() else 'N/A'}")
                    return {
                        **state,
                        "structured_data": None,
                        "error_message": f"Extraction échouée : {str(e)}",
                        "retry_count": state.get("retry_count", 0) + 1
                    }
        
        # Chemin standard (with_structured_output + fallbacks)
        structured_llm = llm.with_structured_output(schema_class)
        start_time = datetime.now()
        
        with OLLAMA_LOCK:
            try:
                result = structured_llm.invoke(extract_messages)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"✅ Extraction terminée en {elapsed:.2f}s")
                
                # Convertir en dict pour l'état
                structured_data = result.model_dump()
                structured_data["fichier_source"] = file_name
                
                # Normaliser les données pour corriger les erreurs de format (listes)
                structured_data = normalize_extracted_data(structured_data, supplier)
                
                # Ré-validater avec le schéma après normalisation
                if supplier == "vitraglass":
                    validated = VitraglassInvoiceSchema(**structured_data)
                    structured_data = validated.model_dump()
                elif supplier == "soprofen":
                    validated = ProfermInvoiceSchema(**structured_data)
                    structured_data = validated.model_dump()
                
                return {
                    **state,
                    "structured_data": structured_data,
                    "error_message": None
                }
            
            except Exception as parse_error:
                # Si l'erreur vient de la validation Pydantic, essayer de récupérer les données brutes
                logger.warning(f"⚠️ Erreur de validation, tentative de récupération des données brutes...")
                
                # Essayer d'extraire le JSON directement depuis le prompt
                try:
                    # Fallback : utiliser le LLM sans structured output pour récupérer le JSON brut
                    raw_llm = get_llm(model=MODEL, format_json=True)
                    result_raw = raw_llm.invoke(extract_messages)
                    raw_data = _extract_json_from_text(result_raw.content or "")
                    
                    # Restructurer si nécessaire
                    raw_data = restructure_json_data(raw_data, supplier, doc_type)
                    raw_data["fichier_source"] = file_name
                    raw_data = normalize_extracted_data(raw_data, supplier)
                    
                    # Ré-essayer la validation
                    if supplier == "vitraglass":
                        validated = VitraglassInvoiceSchema(**raw_data)
                        structured_data = validated.model_dump()
                    elif supplier == "soprofen":
                        validated = ProfermInvoiceSchema(**raw_data)
                        structured_data = validated.model_dump()
                    else:
                        structured_data = raw_data
                    
                    logger.info(f"✅ Extraction récupérée après normalisation")
                    
                    return {
                        **state,
                        "structured_data": structured_data,
                        "error_message": None
                    }
                    
                except Exception as recovery_error:
                    # Certains modèles (ex. glm-ocr) peuvent renvoyer 500 avec format="json"
                    logger.warning(f"⚠️ Premier fallback échoué ({recovery_error}), essai sans format=json...")
                    try:
                        raw_llm_no_fmt = get_llm(model=MODEL)
                        result_raw = raw_llm_no_fmt.invoke(extract_messages)
                        raw_data = _extract_json_from_text(result_raw.content or "")
                        
                        # Restructurer si nécessaire
                        raw_data = restructure_json_data(raw_data, supplier, doc_type)
                        raw_data["fichier_source"] = file_name
                        raw_data = normalize_extracted_data(raw_data, supplier)
                        if supplier == "vitraglass":
                            validated = VitraglassInvoiceSchema(**raw_data)
                            structured_data = validated.model_dump()
                        elif supplier == "soprofen":
                            validated = ProfermInvoiceSchema(**raw_data)
                            structured_data = validated.model_dump()
                        else:
                            structured_data = raw_data
                        logger.info(f"✅ Extraction récupérée (sans format=json)")
                        return {
                            **state,
                            "structured_data": structured_data,
                            "error_message": None
                        }
                    except Exception as fallback2_error:
                        logger.error(f"❌ Échec de la récupération : {str(fallback2_error)}")
                        raise parse_error

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction : {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        return {
            **state,
            "structured_data": None,
            "error_message": f"Extraction échouée : {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1
        }


def validate_node(state: AgentState) -> AgentState:
    """
    Nœud 4 : Vérification visuelle des données extraites (points d'ancrage textuels).
    """
    try:
        structured_data = state["structured_data"]
        doc_type = state["doc_type"]
        supplier = state.get("supplier", "inconnu")
        
        if not structured_data:
            logger.warning("⚠️  Pas de données à valider")
            return {**state, "is_valid": False}
        
        logger.info(f"🔎 Validation des données ({doc_type})")
        is_valid = True
        
        if doc_type == "facture":
            entete = structured_data.get("entete", {})
            lignes = structured_data.get("lignes", [])
            
            if not entete.get("numero_facture"):
                logger.warning("⚠️  Numéro de facture manquant")
                is_valid = False
            
            if not lignes:
                logger.warning("⚠️  Aucune ligne de facturation")
                is_valid = False
            
            # Filtrer les lignes éco-contribution (polluent l'extraction)
            lignes_filtrees = [
                L for L in lignes
                if not (L.get("designation") or "").lower().count("éco-contribution")
                and not (L.get("designation") or "").lower().count("eco-contribution")
            ]
            if len(lignes_filtrees) < len(lignes):
                structured_data["lignes"] = lignes_filtrees
                logger.info(f"   Lignes éco-contribution exclues : {len(lignes) - len(lignes_filtrees)}")
            
            if supplier == "soprofen":
                for i, line in enumerate(lignes_filtrees):
                    ref = (line.get("reference_soi") or "").strip()
                    total_ht = line.get("total_ht")
                    remise = line.get("remise_pourcentage")
                    if ref.upper().startswith("SOI") and (total_ht is None or (isinstance(total_ht, (int, float)) and total_ht <= 0)):
                        logger.warning(f"⚠️  Ligne {i+1} : référence SOI sans Total EUR : {ref[:50]}...")
                        is_valid = False
                    # remise_pourcentage doit être entre 0 et 100 (sinon colonnes mélangées)
                    if isinstance(remise, (int, float)) and (remise < 0 or remise > 100):
                        logger.warning(f"⚠️  Ligne {i+1} : remise_pourcentage={remise} invalide (doit être 0-100). Colonnes probablement mélangées.")
                        is_valid = False
            
            elif supplier == "vitraglass":
                for i, line in enumerate(lignes_filtrees):
                    hl = line.get("hauteur_largeur") or ""
                    des = (line.get("designation") or "")
                    if hl and not re.match(r"\d+\s*[x*×]\s*\d+", hl):
                        logger.warning(f"⚠️  Ligne {i+1} : hauteur_largeur format invalide : {hl}")
                    if des and "D.V." not in des and "d.v." not in des.lower():
                        logger.warning(f"⚠️  Ligne {i+1} : designation sans D.V.")
        
        else:  # devis
            entete = structured_data.get("entete", {})
            if not entete.get("numero_devis"):
                logger.warning("⚠️  Numéro de devis manquant")
                is_valid = False
        
        if is_valid:
            logger.info("✅ Validation réussie")
        else:
            logger.warning("⚠️  Validation échouée, mais on continue...")
            is_valid = True
        
        return {**state, "structured_data": structured_data, "is_valid": is_valid}
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation : {str(e)}")
        return {**state, "is_valid": True}  # On accepte quand même pour ne pas bloquer


def store_db_node(state: AgentState) -> AgentState:
    """
    Nœud 6 : Stocke les données extraites dans la base de données MongoDB.
    """
    try:
        structured_data = state.get("structured_data")
        doc_type = state.get("doc_type")
        supplier = state.get("supplier", "inconnu")
        file_name = state.get("file_name", "")
        
        if not structured_data:
            logger.warning("⚠️  Pas de données à stocker en BDD")
            return state
            
        logger.info(f"🗄️ Stockage en BDD ({doc_type}, {supplier})")
        
        # Appel de la fonction d'insertion
        insert_document(supplier, doc_type, structured_data)
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du stockage en BDD : {str(e)}")
        # On ne bloque pas le flux si le stockage échoue
        return state


def should_retry(state: AgentState) -> Literal["extract", "store_db"]:
    """
    Fonction de routage conditionnel : retry si l'extraction a échoué et retry_count < 3.
    Ne retry pas si le partitionnement a échoué (doc_markdown est None).
    """
    retry_count = state.get("retry_count", 0)
    is_valid = state.get("is_valid", False)
    doc_markdown = state.get("doc_markdown")
    
    # Si le partitionnement a échoué, ne pas retry
    if doc_markdown is None:
        logger.error("❌ Partitionnement échoué, impossible de continuer")
        return "store_db"
    
    # Retry uniquement si validation échouée et retry_count < 3
    if not is_valid and retry_count < 3:
        logger.warning(f"🔄 Nouvelle tentative d'extraction ({retry_count + 1}/3)")
        return "extract"
    else:
        return "store_db"


# ============================================================================
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ============================================================================

def build_graph():
    """Construit et compile le graphe LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Ajouter les nœuds
    workflow.add_node("partition", partition_node)
    workflow.add_node("router_supplier", router_supplier_node)
    workflow.add_node("reformat_markdown", reformat_markdown_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("store_db", store_db_node)
    
    # Définir le point d'entrée
    workflow.set_entry_point("partition")
    
    # Définir les transitions
    workflow.add_edge("partition", "router_supplier")
    workflow.add_edge("router_supplier", "reformat_markdown")
    workflow.add_edge("reformat_markdown", "extract")
    workflow.add_edge("extract", "validate")
    
    # Transition conditionnelle : retry ou store_db
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "extract": "extract",
            "store_db": "store_db"
        }
    )
    
    # Fin du graphe
    workflow.add_edge("store_db", END)
    
    return workflow.compile()


# ============================================================================
# TRAITEMENT ASYNCHRONE D'UN FICHIER
# ============================================================================

def process_single_file(app, pdf_path: str, file_index: int, total_files: int) -> tuple[str, bool, float, Optional[str]]:
    """
    Traite un seul fichier PDF de manière synchrone.
    Retourne: (nom_fichier, succès, durée, message_erreur)
    """
    pdf_name = os.path.basename(pdf_path)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Fichier {file_index}/{total_files} : {pdf_name}")
    logger.info("=" * 60)
    
    # État initial pour ce fichier
    initial_state: AgentState = {
        "file_path": pdf_path,
        "file_name": pdf_name,
        "doc_markdown": None,
        "doc_type": None,
        "supplier": None,
        "table_structure": None,
        "structured_data": None,
        "is_valid": False,
        "retry_count": 0,
        "error_message": None
    }
    
    try:
        # Exécuter le graphe
        start_time = datetime.now()
        final_state = app.invoke(initial_state)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if final_state.get("structured_data"):
            logger.info(f"✅ Traitement terminé avec succès en {elapsed:.2f}s : {pdf_name}")
            return (pdf_name, True, elapsed, None)
        else:
            error_msg = final_state.get("error_message", "Pas de données structurées extraites")
            logger.error(f"❌ Échec du traitement : {pdf_name}")
            logger.error(f"   Erreur : {error_msg}")
            return (pdf_name, False, elapsed, error_msg)
    
    except Exception as e:
        elapsed = 0
        error_msg = str(e)
        logger.error(f"❌ Erreur critique lors du traitement de {pdf_name} : {error_msg}")
        import traceback
        logger.debug(traceback.format_exc())
        return (pdf_name, False, elapsed, error_msg)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def is_file_processed(pdf_name: str) -> bool:
    """
    Vérifie si un fichier PDF a déjà été traité en cherchant dans la base de données.
    """
    return is_document_exists(pdf_name)


def get_pending_pdf_files() -> List[str]:
    """
    Récupère la liste des fichiers PDF dans input/ qui n'ont pas encore été traités.
    """
    all_pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    pending = [pdf for pdf in all_pdfs if not is_file_processed(os.path.basename(pdf))]
    return sorted(pending)


def process_pdf_files(app, pdf_files: List[str]) -> tuple[int, int, float]:
    """
    Traite une liste de fichiers PDF et retourne (succès, échecs, durée).
    """
    if not pdf_files:
        return (0, 0, 0.0)
    
    total_files = len(pdf_files)
    logger.info(f"📚 {total_files} fichier(s) PDF à traiter")
    
    # Traitement parallèle avec ThreadPoolExecutor
    results = []
    start_total = datetime.now()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Soumettre toutes les tâches
        future_to_file = {
            executor.submit(process_single_file, app, pdf_path, idx + 1, total_files): pdf_path
            for idx, pdf_path in enumerate(pdf_files)
        }
        
        # Collecter les résultats au fur et à mesure
        for future in as_completed(future_to_file):
            pdf_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                pdf_name = os.path.basename(pdf_path)
                logger.error(f"❌ Exception non gérée pour {pdf_name} : {str(e)}")
                results.append((pdf_name, False, 0, str(e)))
    
    # Résumé
    elapsed_total = (datetime.now() - start_total).total_seconds()
    successful = sum(1 for _, success, _, _ in results if success)
    failed = total_files - successful
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✨ Traitement terminé ! {total_files} fichier(s) traité(s) en {elapsed_total:.2f}s")
    logger.info(f"   ✅ Succès : {successful}")
    if failed > 0:
        logger.info(f"   ❌ Échecs : {failed}")
    logger.info(f"   ⚡ Parallélisme : {MAX_WORKERS} worker(s)")
    logger.info("=" * 60)
    
    return (successful, failed, elapsed_total)


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def start_fastapi():
    """Démarre le serveur FastAPI dans un thread séparé."""
    import uvicorn
    from src.api import app as fastapi_app
    
    logger.info("🚀 Démarrage du serveur FastAPI sur le port 8000...")
    try:
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de FastAPI : {str(e)}")


def main():
    logger.info("=" * 60)
    logger.info(f"Démarrage avec LangGraph + PDF Native/OCR + glm-ocr")
    logger.info(f"Modèle : {MODEL}")
    logger.info(f"Répertoire d'entrée : {INPUT_DIR}")
    logger.info(f"URL Ollama : {OLLAMA_BASE_URL}")
    logger.info(f"Parallélisme : {MAX_WORKERS} fichier(s) simultané(s)")
    logger.info("=" * 60)
    
    # Construire le graphe (une seule instance partagée)
    logger.info("🔧 Construction du graphe LangGraph...")
    app = build_graph()
    logger.info("✅ Graphe compilé avec succès")
    logger.info("")
    
    # Ensemble pour suivre les fichiers en cours de traitement (thread-safe)
    files_in_progress = set()
    global _files_in_progress_ref
    _files_in_progress_ref = files_in_progress  # partagé avec l'API upload
    
    # Démarrer FastAPI dans un thread séparé
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    logger.info("✅ Thread FastAPI démarré")
    logger.info("")
    logger.info("✅ Serveur prêt - Utilisez l'interface web pour traiter les documents")
    logger.info("")
    
    try:
        # Maintenir le processus actif (FastAPI tourne en daemon)
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("🛑 Arrêt demandé par l'utilisateur")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
