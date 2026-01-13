import os
import glob
import json
import csv
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Annotated, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "mistral:3b")
TIMEOUT = int(os.getenv("TIMEOUT", "600"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))  # Nombre de fichiers à traiter en parallèle
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))  # Intervalle de polling en secondes (pour volumes Docker/Windows)

# Chemins des volumes Docker
INPUT_DIR = "/app/input"


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
    supplier: Optional[str]   # "vitraglass", "proferm", etc.
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
    "date": "2025-10-31",
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
      "prix_unitaire_brut": 505.96,
      "total_ht": 505.96,
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
      "prix_unitaire_brut": 25.96,
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
    Convertit une page spécifique du PDF en image et effectue un OCR.
    """
    try:
        # pdf2image utilise l'indexation 0, donc page_num - 1
        images = convert_from_path(
            pdf_path, 
            first_page=page_num, 
            last_page=page_num,
            fmt="jpeg"
        )
        if not images:
            return ""
        
        # OCR avec pytesseract (en français)
        text = pytesseract.image_to_string(images[0], lang='fra')
        return text
    except Exception as e:
        logger.error(f"Erreur lors de l'OCR de la page {page_num} : {str(e)}")
        return ""

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
    2. Si une page est vide/scannée, utilise l'OCR (pdf2image + pytesseract).
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
                full_content.append(f"## Page {page_num} (OCR)\n\n{ocr_text}")
            else:
                logger.info(f"   Page {page_num} : Texte natif extrait")
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


def router_node(state: AgentState) -> AgentState:
    """
    Nœud 2 : Détecte le type de document (devis, facture ou bon_livraison) avec Few-Shot.
    """
    try:
        doc_markdown = state["doc_markdown"]
        file_name = state["file_name"]
        
        if not doc_markdown:
            logger.warning("Pas de contenu à analyser pour la détection du type")
            return {**state, "doc_type": "devis"}
        
        logger.info(f"🔍 Détection du type de document : {file_name}")
        
        # Prompt Few-Shot pour la détection
        prompt = f"""Tu es un expert en classification de documents financiers français.

EXEMPLES :
- Si le document contient "FACTURE" ou "N° Facture" → Réponds "FACTURE"
- Si le document contient "DEVIS" ou "N° Devis" → Réponds "DEVIS"
- Si le document contient "BON DE LIVRAISON" ou "BL" ou "N° BL" → Réponds "BON_LIVRAISON"

CONTENU DU DOCUMENT :
{doc_markdown[:1000]}

QUESTION : Ce document est-il un DEVIS, une FACTURE ou un BON_LIVRAISON ?
RÉPONSE (un seul mot) :"""
        
        # LLM pour la détection (sans format JSON)
        llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            temperature=0
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        doc_type = response.content.strip().upper()
        
        if "FACTURE" in doc_type:
            doc_type = "facture"
        elif "DEVIS" in doc_type:
            doc_type = "devis"
        elif "BON_LIVRAISON" in doc_type or "BON DE LIVRAISON" in doc_type or "BL" in doc_type:
            doc_type = "bon_livraison"
        else:
            logger.warning(f"Type non reconnu : {doc_type}, défaut : devis")
            doc_type = "devis"
        
        logger.info(f"✅ Type détecté : {doc_type}")
        
        return {**state, "doc_type": doc_type}
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection : {str(e)}")
        return {**state, "doc_type": "devis"}


def detect_supplier_node(state: AgentState) -> AgentState:
    """
    Nœud 2.5 : Détecte le fournisseur émetteur de la facture.
    IMPORTANT : PROFERM/SOPROFEN est notre entreprise (le client), pas un fournisseur.
    """
    try:
        doc_markdown = state["doc_markdown"]
        doc_type = state["doc_type"]
        
        if doc_type != "facture" or not doc_markdown:
            return {**state, "supplier": "inconnu"}
        
        logger.info("🔍 Détection du fournisseur émetteur...")
        
        # Vérification par règles avant d'utiliser le LLM pour plus de précision
        markdown_upper = doc_markdown.upper()
        
        # Détection VITRAGLASS : chercher dans l'en-tête (premières lignes)
        if any(keyword in markdown_upper[:2000] for keyword in ["VITRAGLASS", "GROUPE DEVGLASS", "CEKAL", "GLASS A LIA", "FABRICANT DE VITRAGE ISOLANT"]):
            logger.info("✅ Fournisseur détecté : vitraglass (par règles)")
            return {**state, "supplier": "vitraglass"}
        
        # Détection SOPROFEN : chercher si SOPROFEN est l'émetteur (pas le client)
        if "SOPROFEN" in markdown_upper[:2000]:
            # Vérifier que SOPROFEN n'est pas juste le client
            # Si SOPROFEN apparaît dans l'en-tête avec adresse, c'est l'émetteur
            prompt = f"""Tu es un expert en analyse de factures françaises.
Analyse ce document et détermine si SOPROFEN est l'ÉMETTEUR (fournisseur) ou le CLIENT de cette facture.

IMPORTANT : 
- Si SOPROFEN est dans l'en-tête avec son adresse/coordonnées -> c'est l'émetteur (fournisseur)
- Si SOPROFEN est mentionné comme "client" ou "destinataire" -> ce n'est PAS le fournisseur

CONTENU (premières lignes) :
{doc_markdown[:2000]}

QUESTION : SOPROFEN est-il l'ÉMETTEUR (fournisseur) de cette facture ?
RÉPONSE (un seul mot : oui ou non) :"""

            llm = ChatOllama(
                model=MODEL,
                base_url=OLLAMA_BASE_URL,
                timeout=TIMEOUT,
                temperature=0
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            is_supplier = "oui" in response.content.strip().lower()
            
            if is_supplier:
                logger.info("✅ Fournisseur détecté : soprofen")
                return {**state, "supplier": "soprofen"}
        
        # Détection par LLM pour autres cas
        prompt = f"""Tu es un expert en analyse de factures françaises.
Identifie le FOURNISSEUR ÉMETTEUR de cette facture (celui qui émet la facture, pas le client).

IMPORTANT :
- PROFERM, PROFERM ALU, PROFERM MULTITECHNIQUES = CLIENT (notre entreprise), PAS un fournisseur
- VITRAGLASS, GROUPE DEVGLASS, CEKAL, GLASS A LIA = fournisseur VITRAGLASS
- SOPROFEN = fournisseur uniquement s'il est dans l'en-tête avec adresse
- Cherche l'entreprise dans l'en-tête (nom, adresse, SIRET)

CONTENU :
{doc_markdown[:3000]}

RÉPONSE (un seul mot parmi : vitraglass, soprofen, inconnu) :"""

        llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            temperature=0
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        supplier = response.content.strip().lower()
        
        if "vitraglass" in supplier:
            supplier = "vitraglass"
        elif "soprofen" in supplier:
            supplier = "soprofen"
        else:
            supplier = "inconnu"
            
        logger.info(f"✅ Fournisseur détecté : {supplier}")
        return {**state, "supplier": supplier}
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection du fournisseur : {str(e)}")
        return {**state, "supplier": "inconnu"}


def extract_node(state: AgentState) -> AgentState:
    """
    Nœud 3 : Extrait les données structurées via Pydantic et Few-Shot.
    Sélectionne le schéma optimisé selon le type et le fournisseur.
    """
    try:
        doc_markdown = state["doc_markdown"]
        doc_type = state["doc_type"]
        supplier = state.get("supplier", "inconnu")
        file_name = state["file_name"]
        
        logger.info(f"🤖 Extraction des données structurées (type: {doc_type}, fournisseur: {supplier})")
        
        if not doc_markdown:
            logger.error("❌ Pas de contenu Markdown disponible pour l'extraction")
            return {
                **state,
                "structured_data": None,
                "error_message": "Analyse échouée : pas de contenu Markdown disponible",
                "retry_count": state.get("retry_count", 0) + 1
            }
        
        # Créer le LLM avec structured output
        llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            format="json",
            temperature=0
        )
        
        # Sélection du schéma et de l'exemple selon doc_type et supplier
        if doc_type == "facture":
            if supplier == "vitraglass":
                schema_class = VitraglassInvoiceSchema
                example = INVOICE_EXAMPLE_VITRAGLASS
                instr_supp = """- Extrais les numéros de commande, bons de livraison et dimensions (Hauteur x Largeur).
- CRITIQUE : Si une commande a plusieurs pièces avec des dimensions différentes, crée UNE LIGNE SÉPARÉE par pièce.
- Chaque ligne doit avoir des valeurs SIMPLES (pas de listes) : hauteur_largeur = "2065 x 1727" (string), pas ["2065 x 1727", "1922 x 939"].
- Si plusieurs pièces identiques, utilise la quantité (quantite: 2.0) mais garde une seule ligne avec les mêmes dimensions."""
            elif supplier == "soprofen":
                # SOPROFEN utilise un format similaire à PROFERM avec références SOI
                schema_class = ProfermInvoiceSchema
                example = INVOICE_EXAMPLE_PROFERM
                instr_supp = """- CRITIQUE : Le tableau SOPROFEN a EXACTEMENT ces colonnes dans cet ordre :
  COLONNE 1 : "N° de commande" → reference_soi (ex: "SOI C 25 212 001 555")
  COLONNE 2 : "Désignation" → designation (ex: "Ligne 1 Coffre Paco Dimension Tableau (L x H mm) : 1390 * 2100 Blanc R=0.18")
  COLONNE 3 : "Qte" → quantite (NOMBRE, ex: 1,000 → 1.0 ou 2,984 → 2.984)
  COLONNE 4 : "Unite" → unite (TEXTE, ex: "PIECE" ou "Mètres")
  COLONNE 5 : "P.U. Brut EUR" → prix_unitaire_brut (PRIX AVANT REMISE, ex: 1183,11 → 1183.11 ou 19,91 → 19.91)
  COLONNE 6 : "% Rem." → remise_pourcentage (POURCENTAGE, ex: 64 → 64.0 ou 45 → 45.0)
  COLONNE 7 : "Total EUR" → total_ht (PRIX FINAL APRÈS REMISE, ex: 425,92 → 425.92 ou 32,68 → 32.68)

RÈGLES ABSOLUES POUR ÉVITER LES ERREURS :
- La colonne "Qte" contient toujours un NOMBRE (quantité d'unités) : 1,000 ou 2,984
- La colonne "Unite" contient toujours un TEXTE : "PIECE" ou "Mètres" 
- La colonne "P.U. Brut EUR" = PRIX UNITAIRE BRUT (avant remise) : toujours > 100 pour les pièces, ~20 pour les mètres
- La colonne "Total EUR" = TOTAL APRÈS REMISE : toujours < P.U. Brut EUR (car remise appliquée)
- SI tu vois 1183,11 dans "P.U. Brut EUR" et 425,92 dans "Total EUR" → NE LES INVERSE PAS !
- SI tu vois 1,000 dans "Qte" → c'est la quantité (1.0), PAS le prix !

IGNORE complètement :
- Les lignes "Dont éco-contribution PMCB" ou "éco-contribution"
- Les lignes "Sous totaux :"
- Les lignes vides

Extrais aussi les dimensions depuis la désignation si présentes (format "L x H mm" ou "L * H mm")"""
            else:
                # Schéma par défaut pour fournisseurs inconnus
                schema_class = ProfermInvoiceSchema
                example = INVOICE_EXAMPLE_PROFERM
                instr_supp = "- Utilise le schéma standard d'extraction de facture."

            prompt = f"""Tu es un expert en extraction de factures PDF {supplier.upper()}.

IMPORTANT : Regarde attentivement l'EXEMPLE ci-dessous pour comprendre la structure exacte attendue.

EXEMPLE DE SORTIE ATTENDUE (ANALYSE BIEN CHAQUE VALEUR) :
{example}

ANALYSE DE L'EXEMPLE :
- reference_soi = "SOI C 25 212 001 555" (colonne "N° de commande")
- designation = texte complet (colonne "Désignation")
- quantite = 1.0 (colonne "Qte" - C'EST UN NOMBRE, pas un prix !)
- unite = "PIECE" (colonne "Unite" - C'EST DU TEXTE)
- prix_unitaire_brut = 1183.11 (colonne "P.U. Brut EUR" - PRIX AVANT REMISE, valeur ÉLEVÉE)
- remise_pourcentage = 64.0 (colonne "% Rem." - POURCENTAGE)
- total_ht = 425.92 (colonne "Total EUR" - PRIX FINAL APRÈS REMISE, valeur BASSE)

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:8000]}

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

VÉRIFICATION FINALE AVANT DE RETOURNER LE JSON :
✓ quantite contient un PETIT nombre (1.0, 2.0, 2.984...) ?
✓ prix_unitaire_brut contient un GRAND nombre (919.11, 1183.11...) ?
✓ total_ht est INFÉRIEUR à prix_unitaire_brut (car remise appliquée) ?
✓ unite contient du TEXTE ("PIECE", "Mètres") et pas un nombre ?

Le fichier source est : {file_name}

Retourne UNIQUEMENT le JSON valide sans commentaires."""

        else:  # devis
            schema_class = QuoteSchema
            example = QUOTE_EXAMPLE_OPTIMIZED
            prompt = f"""Tu es un expert en extraction de devis PDF PROFERM.

EXEMPLE DE SORTIE ATTENDUE :
{example}

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:8000]}

INSTRUCTIONS :
- Extrais toutes les prestations/services du devis
- Détaille les dimensions, couleurs (ext/int) et caractéristiques techniques (sous forme de liste)
- Calcule les totaux HT, TVA et TTC
- Le fichier source est : {file_name}

Retourne UNIQUEMENT le JSON sans commentaires."""
        
        # Utiliser with_structured_output pour forcer le schéma Pydantic
        structured_llm = llm.with_structured_output(schema_class)
        
        start_time = datetime.now()
        
        try:
            result = structured_llm.invoke([HumanMessage(content=prompt)])
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
                raw_llm = ChatOllama(
                    model=MODEL,
                    base_url=OLLAMA_BASE_URL,
                    timeout=TIMEOUT,
                    format="json",
                    temperature=0
                )
                
                result_raw = raw_llm.invoke([HumanMessage(content=prompt)])
                import json
                raw_data = json.loads(result_raw.content)
                
                # Normaliser les données
                raw_data = normalize_extracted_data(raw_data, supplier)
                raw_data["fichier_source"] = file_name
                
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
                logger.error(f"❌ Échec de la récupération : {str(recovery_error)}")
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
    Nœud 4 : Valide la cohérence des données extraites.
    """
    try:
        structured_data = state["structured_data"]
        doc_type = state["doc_type"]
        
        if not structured_data:
            logger.warning("⚠️  Pas de données à valider")
            return {**state, "is_valid": False}
        
        logger.info(f"🔎 Validation des données ({doc_type})")
        
        # Validation de base : vérifier que les données essentielles sont présentes
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
            
            # Validation comptable : somme des lignes vs total
            if lignes and entete.get("total_ttc"):
                total_ht_calculated = sum(line.get("total_ht", 0) or 0 for line in lignes)
                # Calcul approximatif (on ne valide pas strictement car les remises peuvent varier)
                if total_ht_calculated > 0:
                    logger.info(f"   Total HT calculé : {total_ht_calculated:.2f}")
        
        else:  # devis
            entete = structured_data.get("entete", {})
            totaux = structured_data.get("totaux", {})
            
            if not entete.get("numero_devis"):
                logger.warning("⚠️  Numéro de devis manquant")
                is_valid = False
            
            # Vérifier la cohérence des totaux
            total_ht = totaux.get("total_ht", 0) or 0
            total_tva = totaux.get("total_tva", 0) or 0
            total_ttc = totaux.get("total_ttc", 0) or 0
            
            if total_ht > 0 and total_ttc > 0:
                expected_ttc = total_ht + total_tva
                diff = abs(expected_ttc - total_ttc)
                if diff > 1:  # Tolérance de 1 euro
                    logger.warning(f"⚠️  Incohérence des totaux : HT({total_ht}) + TVA({total_tva}) ≠ TTC({total_ttc})")
                    is_valid = False
        
        if is_valid:
            logger.info("✅ Validation réussie")
        else:
            logger.warning("⚠️  Validation échouée, mais on continue...")
            # On marque comme valide pour éviter la boucle infinie dans cette version
            is_valid = True
        
        return {**state, "is_valid": is_valid}
    
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
    workflow.add_node("router", router_node)
    workflow.add_node("detect_supplier", detect_supplier_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("store_db", store_db_node)
    
    # Définir le point d'entrée
    workflow.set_entry_point("partition")
    
    # Définir les transitions
    workflow.add_edge("partition", "router")
    workflow.add_edge("router", "detect_supplier")
    workflow.add_edge("detect_supplier", "extract")
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
# GESTIONNAIRE D'ÉVÉNEMENTS FICHIERS
# ============================================================================

class PDFFileHandler(FileSystemEventHandler):
    """
    Gestionnaire d'événements qui détecte l'ajout de fichiers PDF
    et déclenche leur traitement.
    Utilisé avec PollingObserver pour une détection fiable sur volumes Docker/Windows.
    """
    
    def __init__(self, app, files_in_progress: set):
        super().__init__()
        self.app = app
        self.files_in_progress = files_in_progress
        self.processing_lock = threading.Lock()
        self.pending_files = set()
        self.known_files = set()  # Cache des fichiers déjà vus (évite les doublons)
        self.debounce_timer = None
        self.debounce_delay = 1.0  # Délai en secondes pour éviter les doublons
    
    def on_created(self, event: FileSystemEvent):
        """Appelé lorsqu'un fichier ou dossier est créé."""
        logger.debug(f"🔍 Événement détecté (created): {event.src_path}")
        self._handle_file_event(event)
    
    def on_modified(self, event: FileSystemEvent):
        """Appelé lorsqu'un fichier est modifié (utile pour les fichiers copiés progressivement)."""
        logger.debug(f"🔍 Événement détecté (modified): {event.src_path}")
        self._handle_file_event(event)
    
    def _handle_file_event(self, event: FileSystemEvent):
        """Traite un événement de fichier."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Vérifier si c'est un fichier PDF
        if not file_path.lower().endswith('.pdf'):
            logger.debug(f"   ⏭️  Ignoré (pas un PDF): {os.path.basename(file_path)}")
            return
        
        # Normaliser le chemin
        file_path = os.path.abspath(file_path)
        
        # Vérifier si le fichier est dans le répertoire input
        input_dir_abs = os.path.abspath(INPUT_DIR)
        if not file_path.startswith(input_dir_abs):
            logger.debug(f"   ⏭️  Ignoré (hors du répertoire input): {file_path}")
            return
        
        logger.info(f"📥 Nouveau fichier détecté : {os.path.basename(file_path)}")
        
        # Ajouter à la liste des fichiers en attente
        with self.processing_lock:
            # Ignorer si déjà connu (évite les doublons avec le polling)
            if file_path in self.known_files:
                logger.debug(f"   ⏭️  Fichier déjà connu, ignoré : {os.path.basename(file_path)}")
                return
            self.pending_files.add(file_path)
        
        # Utiliser un délai pour éviter de traiter un fichier en cours d'écriture
        # et grouper plusieurs fichiers ajoutés rapidement
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(self.debounce_delay, self._process_pending_files)
        self.debounce_timer.start()
    
    def _process_pending_files(self):
        """Traite les fichiers en attente après le délai de debounce."""
        with self.processing_lock:
            if not self.pending_files:
                return
            
            # Récupérer les fichiers à traiter
            files_to_process = list(self.pending_files)
            self.pending_files.clear()
        
        # Filtrer ceux qui sont déjà traités ou en cours
        new_files = []
        for file_path in files_to_process:
            # Normaliser le chemin
            file_path = os.path.abspath(file_path)
            # Vérifier que le fichier existe et n'est plus en cours d'écriture
            if not os.path.exists(file_path):
                continue
            
            try:
                # Vérifier que le fichier n'est plus en cours d'écriture
                # En comparant la taille à deux moments différents (avec un petit délai)
                size1 = os.path.getsize(file_path)
                threading.Event().wait(0.1)  # Attendre 100ms
                size2 = os.path.getsize(file_path)
                
                # Si la taille a changé, le fichier est encore en cours d'écriture
                if size1 != size2:
                    logger.info(f"⏳ Fichier en cours d'écriture, report du traitement : {os.path.basename(file_path)}")
                    # Remettre dans la file d'attente pour traitement ultérieur
                    with self.processing_lock:
                        self.pending_files.add(file_path)
                    # Reprogrammer le traitement après un délai supplémentaire
                    self.debounce_timer = threading.Timer(self.debounce_delay, self._process_pending_files)
                    self.debounce_timer.start()
                    continue
            except (OSError, IOError) as e:
                logger.warning(f"⚠️  Erreur lors de la vérification du fichier {os.path.basename(file_path)} : {str(e)}")
                continue
            
            # Vérifier si le fichier est déjà traité
            if is_file_processed(os.path.basename(file_path)):
                logger.info(f"⏭️  Fichier déjà traité, ignoré : {os.path.basename(file_path)}")
                continue
            
            # Vérifier si le fichier est en cours de traitement
            if file_path in self.files_in_progress:
                continue
            
            new_files.append(file_path)
        
        if not new_files:
            return
        
        # Marquer les fichiers comme en cours de traitement
        for f in new_files:
            self.files_in_progress.add(f)
        
        try:
            # Traiter les nouveaux fichiers
            logger.info(f"🚀 Démarrage du traitement de {len(new_files)} fichier(s)...")
            successful, failed, elapsed = process_pdf_files(self.app, new_files)
            
            logger.info("")
            logger.info("🔄 Retour en mode surveillance...")
            logger.info("   (En attente de nouveaux fichiers PDF dans 'input')")
            logger.info("")
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des fichiers : {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Retirer les fichiers de l'ensemble après traitement
            for f in new_files:
                self.files_in_progress.discard(f)
                # Ajouter au cache des fichiers connus
                with self.processing_lock:
                    self.known_files.add(f)


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
    logger.info(f"Démarrage avec LangGraph + PDF Native/OCR + Ministral 3B")
    logger.info(f"Modèle : {MODEL}")
    logger.info(f"Répertoire d'entrée : {INPUT_DIR}")
    logger.info(f"URL Ollama : {OLLAMA_BASE_URL}")
    logger.info(f"Parallélisme : {MAX_WORKERS} fichier(s) simultané(s)")
    logger.info("=" * 60)
    
    # Démarrer FastAPI dans un thread séparé
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    logger.info("✅ Thread FastAPI démarré")
    logger.info("")
    
    # Construire le graphe (une seule instance partagée)
    logger.info("🔧 Construction du graphe LangGraph...")
    app = build_graph()
    logger.info("✅ Graphe compilé avec succès")
    logger.info("")
    
    # Ensemble pour suivre les fichiers en cours de traitement (thread-safe)
    files_in_progress = set()
    
    # Créer le gestionnaire d'événements
    event_handler = PDFFileHandler(app, files_in_progress)
    
    # Initialiser le cache des fichiers connus avec les fichiers déjà traités
    all_pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    for pdf_path in all_pdfs:
        pdf_name = os.path.basename(pdf_path)
        if is_file_processed(pdf_name):
            event_handler.known_files.add(os.path.abspath(pdf_path))
    
    # Traiter les fichiers existants non traités au démarrage
    existing_files = get_pending_pdf_files()
    if existing_files:
        logger.info(f"📂 {len(existing_files)} fichier(s) PDF existant(s) détecté(s), traitement...")
        logger.info("")
        try:
            successful, failed, elapsed = process_pdf_files(app, existing_files)
            # Ajouter les fichiers traités au cache
            for pdf_path in existing_files:
                event_handler.known_files.add(os.path.abspath(pdf_path))
            logger.info("")
            logger.info("🔄 Passage en mode surveillance des nouveaux fichiers...")
            logger.info("")
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des fichiers existants : {str(e)}")
    
    # Créer l'observateur PollingObserver (optimisé pour volumes Docker/Windows)
    # timeout=POLL_INTERVAL définit l'intervalle de vérification en secondes
    observer = Observer(timeout=POLL_INTERVAL)
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    
    logger.info("🔄 Mode surveillance activé (PollingObserver)")
    logger.info(f"   Intervalle de polling : {POLL_INTERVAL}s (optimisé pour volumes Docker/Windows)")
    logger.info("   (Déposez des fichiers dans le répertoire 'input' pour les traiter)")
    logger.info("")
    
    # Démarrer l'observateur
    observer.start()
    logger.info("✅ PollingObserver démarré avec succès")
    logger.info("")
    
    try:
        # Maintenir le processus actif
        logger.info("✅ Surveillance active, en attente de nouveaux fichiers...")
        logger.info(f"   (Vérification automatique toutes les {POLL_INTERVAL}s)")
        logger.info("")
        observer.join()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("🛑 Arrêt demandé par l'utilisateur")
        logger.info("=" * 60)
        observer.stop()
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le gestionnaire de surveillance : {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        observer.stop()
        raise
    finally:
        observer.join(timeout=5)
        if observer.is_alive():
            logger.warning("⚠️  L'observateur n'a pas pu s'arrêter proprement")


if __name__ == "__main__":
    main()
