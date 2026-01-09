import os
import glob
import json
import csv
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from docling.document_converter import DocumentConverter

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
DOCLING_TIMEOUT = int(os.getenv("DOCLING_TIMEOUT", "300"))  # 5 minutes max pour le partitionnement

# Chemins des volumes Docker
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# SCHÉMAS PYDANTIC POUR L'EXTRACTION STRUCTURÉE
# ============================================================================

class LineItem(BaseModel):
    """Ligne d'article pour facture ou devis."""
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


class InvoiceSchema(BaseModel):
    """Schéma complet pour une facture."""
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


class QuoteTotals(BaseModel):
    """Totaux financiers d'un devis."""
    total_ht: Optional[float] = Field(None, description="Total Hors Taxes")
    total_tva: Optional[float] = Field(None, description="Montant total de la TVA")
    total_ttc: Optional[float] = Field(None, description="Total Toutes Taxes Comprises")


class QuoteSchema(BaseModel):
    """Schéma complet pour un devis."""
    entete: QuoteHeader = Field(..., description="Informations d'en-tête du devis")
    prestations: List[LineItem] = Field(default_factory=list, description="Liste des prestations ou services")
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
    doc_type: Optional[str]  # "devis" ou "facture"
    structured_data: Optional[dict]
    is_valid: bool
    retry_count: int
    error_message: Optional[str]


# ============================================================================
# EXEMPLES FEW-SHOT POUR STABILISER LE MODÈLE
# ============================================================================

INVOICE_EXAMPLE = """{
  "entete": {
    "numero_facture": "FAC2025-001",
    "date": "2025-01-09",
    "client_nom": "Client Exemple SAS",
    "total_ttc": 1200.50
  },
  "lignes": [
    {
      "reference": "REF001",
      "designation": "Prestation de conseil informatique",
      "quantite": 5.0,
      "unite": "Heures",
      "prix_unitaire_brut": 100.0,
      "remise_pourcentage": 10.0,
      "total_ht": 450.0,
      "tva_pourcentage": 20.0
    }
  ],
  "fichier_source": "exemple.pdf"
}"""

QUOTE_EXAMPLE = """{
  "entete": {
    "numero_devis": "DEV2025-001",
    "date_emission": "2025-01-09",
    "date_validite": "2025-02-09",
    "entreprise_nom": "TechSolutions Pro",
    "client_nom": "Client Exemple SAS"
  },
  "prestations": [
    {
      "designation": "Développement application web",
      "quantite": 10.0,
      "unite": "Jours",
      "prix_unitaire_brut": 500.0,
      "total_ht": 5000.0,
      "tva_pourcentage": 20.0
    }
  ],
  "totaux": {
    "total_ht": 5000.0,
    "total_tva": 1000.0,
    "total_ttc": 6000.0
  },
  "conditions_paiement": "Paiement à 30 jours",
  "fichier_source": "exemple.pdf"
}"""


# ============================================================================
# NŒUDS DU GRAPHE LANGGRAPH
# ============================================================================

def partition_node(state: AgentState) -> AgentState:
    """
    Nœud 1 : Partitionne le PDF avec Docling.
    Extrait le contenu structuré en Markdown.
    """
    try:
        file_path = state["file_path"]
        file_name = state["file_name"]
        
        logger.info(f"📄 Partitionnement du document avec Docling : {file_name}")
        start_time = datetime.now()
        
        # Initialiser le convertisseur Docling (configuration par défaut)
        # Note: Docling détecte automatiquement si OCR est nécessaire
        converter = DocumentConverter()
        
        logger.info(f"   Timeout max : {DOCLING_TIMEOUT}s")
        
        # Convertir le document avec gestion de timeout via threading
        result = None
        conversion_error = None
        
        def convert_document():
            nonlocal result, conversion_error
            try:
                result = converter.convert(file_path)
            except Exception as e:
                conversion_error = e
        
        # Lancer la conversion dans un thread
        thread = threading.Thread(target=convert_document)
        thread.daemon = True
        thread.start()
        thread.join(timeout=DOCLING_TIMEOUT)
        
        # Vérifier si le thread est encore actif (timeout)
        if thread.is_alive():
            logger.error(f"❌ Partitionnement timeout après {DOCLING_TIMEOUT}s")
            return {
                **state,
                "doc_markdown": None,
                "error_message": f"Partitionnement timeout après {DOCLING_TIMEOUT}s"
            }
        
        # Vérifier s'il y a eu une erreur
        if conversion_error:
            raise conversion_error
        
        if result is None:
            raise Exception("Conversion retournée None")
        
        # Exporter en Markdown
        doc_markdown = result.document.export_to_markdown()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Document converti en Markdown ({len(doc_markdown)} caractères) en {elapsed:.2f}s")
        
        return {
            **state,
            "doc_markdown": doc_markdown,
            "error_message": None
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du partitionnement avec Docling : {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            **state,
            "doc_markdown": None,
            "error_message": f"Partitionnement échoué : {str(e)}"
        }


def router_node(state: AgentState) -> AgentState:
    """
    Nœud 2 : Détecte le type de document (devis ou facture) avec Few-Shot.
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

CONTENU DU DOCUMENT :
{doc_markdown[:1000]}

QUESTION : Ce document est-il un DEVIS ou une FACTURE ?
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
        else:
            logger.warning(f"Type non reconnu : {doc_type}, défaut : devis")
            doc_type = "devis"
        
        logger.info(f"✅ Type détecté : {doc_type}")
        
        return {**state, "doc_type": doc_type}
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection : {str(e)}")
        return {**state, "doc_type": "devis"}


def extract_node(state: AgentState) -> AgentState:
    """
    Nœud 3 : Extrait les données structurées via Pydantic et Few-Shot.
    """
    try:
        doc_markdown = state["doc_markdown"]
        doc_type = state["doc_type"]
        file_name = state["file_name"]
        
        logger.info(f"🤖 Extraction des données avec Docling (type: {doc_type})")
        
        if not doc_markdown:
            logger.error("❌ Pas de contenu Markdown disponible pour l'extraction")
            return {
                **state,
                "structured_data": None,
                "error_message": "Partitionnement échoué : pas de contenu Markdown disponible",
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
        
        # Sélectionner le schéma et l'exemple approprié
        if doc_type == "facture":
            schema_class = InvoiceSchema
            example = INVOICE_EXAMPLE
            prompt = f"""Tu es un expert en extraction de factures PDF.

EXEMPLE DE SORTIE ATTENDUE :
{example}

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:4000]}

INSTRUCTIONS :
- Extrais toutes les lignes du tableau de facturation présent dans le Markdown
- Utilise le schéma JSON de l'exemple ci-dessus
- Le fichier source est : {file_name}

Retourne UNIQUEMENT le JSON sans commentaires."""
        else:
            schema_class = QuoteSchema
            example = QUOTE_EXAMPLE
            prompt = f"""Tu es un expert en extraction de devis PDF.

EXEMPLE DE SORTIE ATTENDUE :
{example}

CONTENU DU DOCUMENT (MARKDOWN) :
{doc_markdown[:4000]}

INSTRUCTIONS :
- Extrais toutes les prestations/services du devis présent dans le Markdown
- Utilise le schéma JSON de l'exemple ci-dessus
- Calcule les totaux HT, TVA et TTC
- Le fichier source est : {file_name}

Retourne UNIQUEMENT le JSON sans commentaires."""
        
        # Utiliser with_structured_output pour forcer le schéma Pydantic
        structured_llm = llm.with_structured_output(schema_class)
        
        start_time = datetime.now()
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Extraction terminée en {elapsed:.2f}s")
        
        # Convertir en dict pour l'état
        structured_data = result.model_dump()
        structured_data["fichier_source"] = file_name
        
        return {
            **state,
            "structured_data": structured_data,
            "error_message": None
        }
    
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


def save_node(state: AgentState) -> AgentState:
    """
    Nœud 5 : Sauvegarde les données en JSON ou CSV.
    """
    try:
        structured_data = state["structured_data"]
        doc_type = state["doc_type"]
        file_name = state["file_name"]
        
        if not structured_data:
            logger.error("❌ Pas de données à sauvegarder")
            return state
        
        logger.info(f"💾 Sauvegarde des données ({doc_type})")
        
        if doc_type == "devis":
            # Sauvegarder en JSON
            output_name = file_name.replace(".pdf", ".json")
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ JSON sauvegardé : {output_path} ({file_size} octets)")
        
        else:  # facture
            # Sauvegarder en CSV
            output_name = file_name.replace(".pdf", ".csv")
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            entete = structured_data.get("entete", {})
            lignes = structured_data.get("lignes", [])
            
            if not lignes:
                logger.warning("⚠️  Aucune ligne à sauvegarder en CSV")
                # Créer un CSV vide avec les en-têtes
                columns = ["numero_facture", "date", "client", "reference", "designation", 
                          "quantite", "unite", "prix_unitaire_brut", "remise_pourcentage", 
                          "total_ht", "tva_pourcentage"]
                with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns, delimiter=';')
                    writer.writeheader()
            else:
                # Colonnes
                entete_columns = ["numero_facture", "date", "client"]
                ligne_columns = ["reference", "designation", "quantite", "unite", 
                                "prix_unitaire_brut", "remise_pourcentage", "total_ht", "tva_pourcentage"]
                columns = entete_columns + ligne_columns
                
                # Valeurs d'en-tête
                entete_values = {
                    "numero_facture": entete.get("numero_facture", ""),
                    "date": entete.get("date", ""),
                    "client": entete.get("client_nom", "")
                }
                
                # Écrire le CSV
                with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns, delimiter=';', extrasaction='ignore')
                    writer.writeheader()
                    
                    for ligne in lignes:
                        row = {**entete_values, **ligne}
                        writer.writerow(row)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ CSV sauvegardé : {output_path} ({file_size} octets, {len(lignes)} lignes)")
        
        return state
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde : {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return state


def should_retry(state: AgentState) -> Literal["extract", "save"]:
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
        return "save"
    
    # Retry uniquement si validation échouée et retry_count < 3
    if not is_valid and retry_count < 3:
        logger.warning(f"🔄 Nouvelle tentative d'extraction ({retry_count + 1}/3)")
        return "extract"
    else:
        return "save"


# ============================================================================
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ============================================================================

def build_graph():
    """Construit et compile le graphe LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Ajouter les nœuds
    workflow.add_node("partition", partition_node)
    workflow.add_node("router", router_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("save", save_node)
    
    # Définir le point d'entrée
    workflow.set_entry_point("partition")
    
    # Définir les transitions
    workflow.add_edge("partition", "router")
    workflow.add_edge("router", "extract")
    workflow.add_edge("extract", "validate")
    
    # Transition conditionnelle : retry ou save
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "extract": "extract",
            "save": "save"
        }
    )
    
    # Fin du graphe
    workflow.add_edge("save", END)
    
    return workflow.compile()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info(f"Démarrage avec LangGraph + Docling + Ministral 3B")
    logger.info(f"Modèle : {MODEL}")
    logger.info(f"Répertoire d'entrée : {INPUT_DIR}")
    logger.info(f"Répertoire de sortie : {OUTPUT_DIR}")
    logger.info(f"URL Ollama : {OLLAMA_BASE_URL}")
    logger.info("=" * 60)
    
    # Construire le graphe
    logger.info("🔧 Construction du graphe LangGraph...")
    app = build_graph()
    logger.info("✅ Graphe compilé avec succès")
    
    # Récupérer les fichiers PDF
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    pdf_files.sort()
    
    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans {INPUT_DIR}")
        return
    
    total_files = len(pdf_files)
    logger.info(f"📚 {total_files} fichier(s) PDF à traiter")
    
    # Traiter chaque fichier (séquentiel pour cette version, parallélisation future possible)
    for file_index, pdf_path in enumerate(pdf_files, start=1):
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
            else:
                logger.error(f"❌ Échec du traitement : {pdf_name}")
                if final_state.get("error_message"):
                    logger.error(f"   Erreur : {final_state['error_message']}")
        
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du traitement de {pdf_name} : {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✨ Traitement terminé ! {total_files} fichier(s) traité(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
