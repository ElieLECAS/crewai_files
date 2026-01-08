import os
import glob
import json
import csv
import logging
import base64
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Tuple
import pdfplumber
from pdf2image import convert_from_path
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Configuration du logging avec timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 1. Chargement de l'environnement
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "mistral:7b")
TIMEOUT = int(os.getenv("TIMEOUT", "600"))

# Chemins des volumes Docker
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pdf_text_or_images(pdf_path: str) -> tuple[Optional[str], Optional[List[str]]]:
    """
    Charge un PDF et retourne soit le texte extrait (avec tableaux structurés), soit les images des pages.
    Retourne (texte, None) si texte disponible, (None, [images_base64]) si besoin OCR.
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"Le fichier {pdf_path} n'existe pas")
            return None, None
        
        logger.info(f"Début du chargement PDF : {os.path.basename(pdf_path)}")
        
        # 1. Essayer d'abord l'extraction directe du texte avec tableaux structurés (rapide, sans OCR)
        try:
            logger.info("Tentative d'extraction directe du texte et des tableaux (pdfplumber)...")
            text_content = []
            tables_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extraire le texte normal
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Page {page_num} ---\n{page_text}\n")
                    
                    # Extraire les tableaux avec structure préservée
                    tables = page.extract_tables()
                    if tables:
                        logger.info(f"  📊 {len(tables)} tableau(x) détecté(s) sur la page {page_num}")
                        for table_idx, table in enumerate(tables, start=1):
                            if table and len(table) > 0:
                                # Nettoyer le tableau (supprimer les lignes vides)
                                cleaned_table = [row for row in table if row and any(cell and str(cell).strip() for cell in row)]
                                
                                if cleaned_table:
                                    # Formater le tableau en format Markdown pour meilleure lisibilité
                                    table_text = f"\n{'='*80}\n"
                                    table_text += f"TABLEAU {table_idx} - Page {page_num}\n"
                                    table_text += f"{'='*80}\n\n"
                                    
                                    # Utiliser la première ligne comme en-tête
                                    headers = cleaned_table[0]
                                    # Nettoyer les en-têtes
                                    headers = [str(cell).strip() if cell else f"Colonne_{i+1}" for i, cell in enumerate(headers)]
                                    
                                    # Calculer la largeur de chaque colonne
                                    col_widths = []
                                    for col_idx in range(len(headers)):
                                        max_width = len(headers[col_idx])
                                        for row in cleaned_table[1:]:
                                            if col_idx < len(row) and row[col_idx]:
                                                max_width = max(max_width, len(str(row[col_idx]).strip()))
                                        col_widths.append(max(max_width, 10))  # Minimum 10 caractères
                                    
                                    # En-tête
                                    header_line = " | ".join([str(h).ljust(col_widths[i]) for i, h in enumerate(headers)])
                                    table_text += header_line + "\n"
                                    table_text += "-" * (sum(col_widths) + (len(headers) - 1) * 3) + "\n"
                                    
                                    # Lignes de données
                                    for row in cleaned_table[1:]:
                                        if row:
                                            row_values = []
                                            for col_idx in range(len(headers)):
                                                if col_idx < len(row) and row[col_idx]:
                                                    row_values.append(str(row[col_idx]).strip().ljust(col_widths[col_idx]))
                                                else:
                                                    row_values.append("".ljust(col_widths[col_idx]))
                                            table_text += " | ".join(row_values) + "\n"
                                    
                                    table_text += f"\n{'='*80}\n"
                                    tables_content.append(table_text)
            
            # Combiner texte et tableaux
            full_content = []
            if text_content:
                full_content.append("\n".join(text_content))
            if tables_content:
                full_content.append("\n" + "="*80 + "\n")
                full_content.append("TABLEAUX EXTRAITS (STRUCTURE PRÉSERVÉE):\n")
                full_content.append("="*80 + "\n")
                full_content.append("\n".join(tables_content))
            
            if full_content and len("".join(full_content).strip()) > 50:  # Seuil minimum de texte
                full_text = "\n".join(full_content)
                logger.info(f"✅ Texte et tableaux extraits directement (sans OCR) : {len(full_text)} caractères, {len(text_content)} page(s), {len(tables_content)} tableau(x)")
                return full_text, None
            else:
                logger.warning("Peu ou pas de contenu extrait directement, conversion en images pour OCR LLM...")
        except Exception as e:
            logger.warning(f"Extraction directe échouée ({str(e)}), conversion en images pour OCR LLM...")
        
        # 2. Fallback : Convertir les pages en images pour OCR via LLM
        logger.info("Conversion des pages PDF en images pour OCR via Ministral 3B...")
        try:
            images = convert_from_path(pdf_path, dpi=200)
            if not images:
                logger.warning(f"Aucune image générée du fichier {pdf_path}")
                return None, None
            
            # Convertir les images en base64 pour l'envoi au LLM
            images_base64 = []
            for i, image in enumerate(images, start=1):
                # Sauvegarder temporairement en PNG pour conversion base64
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name, 'PNG')
                    with open(tmp_file.name, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                        images_base64.append(img_base64)
                    os.unlink(tmp_file.name)  # Nettoyer le fichier temporaire
            
            logger.info(f"✅ {len(images_base64)} page(s) convertie(s) en images pour OCR LLM")
            return None, images_base64
        
        except Exception as e:
            logger.error(f"Erreur lors de la conversion en images : {str(e)}")
            return None, None
    
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du PDF {pdf_path}: {str(e)}")
        return None, None


def detect_document_type(pdf_text: Optional[str], pdf_images: Optional[List[str]], pdf_name: str, llm: ChatOllama) -> Optional[str]:
    """Détecte le type de document (devis ou facture)."""
    try:
        logger.info(f"Détection du type de document pour : {pdf_name}")
        
        if pdf_text:
            prompt_text = f"""
Analyse ce document et détermine s'il s'agit d'un DEVIS ou d'une FACTURE.

CONTENU DU PDF :
{pdf_text[:2000]}

Réponds UNIQUEMENT par un mot : soit "DEVIS" soit "FACTURE".
"""
            messages = [HumanMessage(content=prompt_text)]
        else:
            prompt_text = f"""
Analyse ce document et détermine s'il s'agit d'un DEVIS ou d'une FACTURE.

Réponds UNIQUEMENT par un mot : soit "DEVIS" soit "FACTURE".
"""
            content = [{"type": "text", "text": prompt_text}]
            for img_base64 in pdf_images[:1]:  # Utiliser seulement la première image pour la détection
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_base64}"
                })
            messages = [HumanMessage(content=content)]
        
        response = llm.invoke(messages)
        doc_type = response.content.strip().upper()
        
        if "DEVIS" in doc_type:
            doc_type = "devis"
        elif "FACTURE" in doc_type:
            doc_type = "facture"
        else:
            logger.warning(f"Type de document non reconnu : {doc_type}, par défaut : devis")
            doc_type = "devis"
        
        logger.info(f"Type de document détecté : {doc_type}")
        return doc_type
    
    except Exception as e:
        logger.error(f"Erreur lors de la détection du type de document : {str(e)}")
        return "devis"  # Par défaut


def extract_data_with_llm(pdf_text: Optional[str], pdf_images: Optional[List[str]], pdf_name: str, doc_type: str, llm: ChatOllama) -> Optional[dict]:
    """Extrait et structure les données du PDF via LangChain Chain avec support vision."""
    try:
        logger.info(f"Début de l'extraction des données avec le LLM (type: {doc_type})")
        
        # Préparer le prompt selon le type de document et le type d'input
        if doc_type == "devis":
            # Format JSON pour les devis
            if pdf_text:
                logger.info(f"Traitement du texte extrait : {len(pdf_text)} caractères")
                prompt_text = f"""
Tu es un expert en extraction de données de devis PDF.
Tu es spécialisé dans l'analyse de documents commerciaux français et l'extraction structurée d'informations financières.

CONTENU DU PDF EXTRAIT :
{pdf_text}

TA MISSION :
Analyse le contenu extrait ci-dessus et structure les données en objet JSON.
Extrais toutes les informations pertinentes que tu trouves dans le document, en organisant les données de la manière la plus logique et structurée possible selon le contenu du document.

Tu peux inclure (mais n'es pas limité à) :
- Le numéro de devis
- Les dates (émission, validité, etc.)
- Les informations de l'entreprise (nom, adresse, coordonnées, etc.)
- Les prestations/services (description, quantité, prix, montants, etc.)
- Les totaux financiers (HT, TVA, TTC, etc.)
- Les conditions de paiement (délai, mode, etc.)
- Toute autre information pertinente trouvée dans le document
- Le nom du fichier source : {pdf_name}

IMPORTANT : 
- Retourne UNIQUEMENT un objet JSON valide.
- Tu es libre de choisir la structure JSON qui correspond le mieux au contenu du document.
- Organise les données de manière logique et cohérente.
- Utilise des noms de champs clairs et descriptifs.
- Si une information n'est pas trouvée, tu peux l'omettre ou utiliser null.
- Les valeurs numériques peuvent être des nombres ou des chaînes selon le contexte.
"""
                messages = [HumanMessage(content=prompt_text)]
            else:
                # Mode OCR avec images
                logger.info(f"Traitement OCR avec {len(pdf_images)} image(s) via vision LLM")
                prompt_text = f"""
Tu es un expert en extraction de données de devis PDF avec capacités OCR.
Analyse les images du document PDF ci-dessous et extrais toutes les informations du devis.

TA MISSION :
À partir des images du PDF, identifie et extrais toutes les informations pertinentes, en organisant les données de la manière la plus logique et structurée possible selon le contenu du document.

Tu peux inclure (mais n'es pas limité à) :
- Le numéro de devis
- Les dates (émission, validité, etc.)
- Les informations de l'entreprise (nom, adresse, coordonnées, etc.)
- Les prestations/services (description, quantité, prix, montants, etc.)
- Les totaux financiers (HT, TVA, TTC, etc.)
- Les conditions de paiement (délai, mode, etc.)
- Toute autre information pertinente trouvée dans le document
- Le nom du fichier source : {pdf_name}

IMPORTANT : 
- Retourne UNIQUEMENT un objet JSON valide.
- Tu es libre de choisir la structure JSON qui correspond le mieux au contenu du document.
- Organise les données de manière logique et cohérente.
- Utilise des noms de champs clairs et descriptifs.
- Si une information n'est pas trouvée, tu peux l'omettre ou utiliser null.
- Les valeurs numériques peuvent être des nombres ou des chaînes selon le contexte.
"""
                # Créer les messages avec images
                content = [{"type": "text", "text": prompt_text}]
                for img_base64 in pdf_images:
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_base64}"
                    })
                messages = [HumanMessage(content=content)]
        else:
            # Format CSV pour les factures
            if pdf_text:
                logger.info(f"Traitement du texte extrait : {len(pdf_text)} caractères")
                prompt_text = f"""
Tu es un expert en extraction de données de factures PDF.
Tu es spécialisé dans l'analyse de documents commerciaux français (Soprofen, etc.) et l'extraction de tableaux de lignes de facturation.

CONTENU DU PDF EXTRAIT :
{pdf_text}

TA MISSION :
Le document contient des TABLEAUX STRUCTURÉS (section "TABLEAUX EXTRAITS" ci-dessous).
Utilise ces tableaux structurés pour extraire les données - ils préservent l'alignement des colonnes.

Identifie le tableau principal des articles/prestations et extrais TOUTES les lignes.
Sois extrêmement vigilant sur l'alignement des colonnes. Ne confonds pas les nombres présents dans la description avec les colonnes de quantité ou de prix.

Pour chaque ligne du tableau structuré, extrais précisément :
- reference : Le numéro de commande ou référence (souvent à gauche)
- designation : La description complète de l'article (fusionne les lignes si la description est sur plusieurs lignes)
- quantite : Le nombre uniquement (ex: 2,984) - Ignore les "1 pièces" ou autres nombres dans la description
- unite : L'unité de mesure (ex: Mètres, Pièces, Unité)
- prix_unitaire_brut : Le prix unitaire avant remise
- remise_pourcentage : Le pourcentage de remise uniquement (ex: 45)
- total_ht : Le montant total HT de la ligne
- tva_pourcentage : Le taux de TVA applicable (ex: 20)

STRUCTURE DE SORTIE JSON :
{{
  "entete": {{
    "numero_facture": "...",
    "date": "...",
    "client_nom": "...",
    "total_ttc": ...
  }},
  "lignes": [
    {{
      "reference": "...",
      "designation": "...",
      "quantite": ...,
      "unite": "...",
      "prix_unitaire_brut": ...,
      "remise_pourcentage": ...,
      "total_ht": ...,
      "tva_pourcentage": ...
    }}
  ]
}}

IMPORTANT : 
- Retourne UNIQUEMENT l'objet JSON.
- Nettoie les valeurs : pas de "€" ou "EUR" dans les champs numériques, utilise le point ou la virgule pour les décimaux.
- Si une colonne est vide, mets null.
- Le nom du fichier source : {pdf_name}
"""
                messages = [HumanMessage(content=prompt_text)]
            else:
                # Mode OCR avec images
                logger.info(f"Traitement OCR avec {len(pdf_images)} image(s) via vision LLM")
                prompt_text = f"""
Tu es un expert en extraction de données de factures PDF par vision (OCR).
Analyse l'image de la facture et extrais le tableau des lignes de facturation avec une précision chirurgicale.

TA MISSION :
1. Repère les colonnes du tableau : N° de commande, Désignation, Qté, Unité, P.U. Brut, % Rem., Total.
2. Pour chaque ligne, extrais les données en faisant attention à ce que chaque valeur appartienne bien à sa colonne.
3. Ne prends pas les nombres à l'intérieur du texte de description pour les quantités ou les prix.

STRUCTURE DE SORTIE JSON :
{{
  "entete": {{
    "numero_facture": "...",
    "date": "...",
    "client_nom": "...",
    "total_ttc": ...
  }},
  "lignes": [
    {{
      "reference": "...",
      "designation": "...",
      "quantite": ...,
      "unite": "...",
      "prix_unitaire_brut": ...,
      "remise_pourcentage": ...,
      "total_ht": ...,
      "tva_pourcentage": ...
    }}
  ]
}}

IMPORTANT : 
- Retourne UNIQUEMENT l'objet JSON.
- Fusionne les descriptions qui s'étalent sur plusieurs lignes pour une même référence.
- Nettoie les valeurs numériques (pas de symboles monétaires).
- Le nom du fichier source : {pdf_name}
"""
                # Créer les messages avec images
                content = [{"type": "text", "text": prompt_text}]
                for img_base64 in pdf_images:
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_base64}"
                    })
                messages = [HumanMessage(content=content)]
        
        # Exécution
        start_time = datetime.now()
        response = llm.invoke(messages)
        
        # Parser la réponse JSON
        response_text = response.content.strip()
        
        # Nettoyer le contenu pour extraire uniquement le JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parser le JSON de manière flexible sans validation de schéma
        try:
            result_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON : {str(e)}")
            logger.debug(f"Contenu reçu : {response_text[:500]}...")
            return None
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Extraction terminée avec succès en {elapsed_time:.2f} secondes")
        
        logger.info(f"Données structurées : {len(json.dumps(result_dict))} caractères JSON")
        
        return result_dict
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des données : {str(e)}")
        import traceback
        logger.debug(f"Traceback : {traceback.format_exc()}")
        return None


def save_json(data: dict, output_path: str) -> bool:
    """Sauvegarde les données en fichier JSON."""
    try:
        logger.info(f"Sauvegarde du fichier JSON : {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Fichier sauvegardé avec succès : {output_path} ({file_size} octets)")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {output_path}: {str(e)}")
        return False


def save_csv(data: dict, output_path: str) -> bool:
    """Sauvegarde les données de facture en fichier CSV propre pour Excel."""
    try:
        logger.info(f"Sauvegarde du fichier CSV : {output_path}")
        
        # Extraire les lignes de facture
        lignes = data.get("lignes", [])
        entete = data.get("entete", {})
        
        if not lignes:
            logger.warning("Aucune ligne de facture trouvée dans les données")
            # Créer un CSV vide avec juste les en-têtes
            columns = ["numero_facture", "date", "client", "reference", "designation", 
                      "quantite", "unite", "prix_unitaire_brut", "remise_pourcentage", "total_ht", "tva_pourcentage"]
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns, delimiter=';')
                writer.writeheader()
            file_size = os.path.getsize(output_path)
            logger.info(f"Fichier CSV sauvegardé (vide) : {output_path} ({file_size} octets)")
            return True
        
        # Colonnes d'en-tête de facture (répétées sur chaque ligne pour faciliter le filtrage Excel)
        entete_columns = ["numero_facture", "date", "client"]
        
        # Colonnes des lignes de facture
        ligne_columns = [
            "reference", "designation", "quantite", "unite", 
            "prix_unitaire_brut", "remise_pourcentage", "total_ht", "tva_pourcentage"
        ]
        
        # Ajouter les colonnes supplémentaires si présentes dans les données
        all_keys = set()
        for ligne in lignes:
            all_keys.update(ligne.keys())
        
        for key in sorted(all_keys):
            if key not in ligne_columns:
                ligne_columns.append(key)
        
        # Colonnes finales
        columns = entete_columns + ligne_columns
        
        # Préparer les valeurs d'en-tête
        entete_values = {
            "numero_facture": entete.get("numero_facture", entete.get("numero", "")),
            "date": entete.get("date", ""),
            "client": entete.get("client_nom", entete.get("client", ""))
        }
        
        # Écrire le CSV (utf-8-sig pour que Excel reconnaisse l'encodage)
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=';', extrasaction='ignore')
            writer.writeheader()
            
            # Écrire les lignes de facture avec les infos d'en-tête
            for ligne in lignes:
                row = {**entete_values, **ligne}
                writer.writerow(row)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Fichier CSV sauvegardé avec succès : {output_path} ({file_size} octets, {len(lignes)} lignes)")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier CSV {output_path}: {str(e)}")
        import traceback
        logger.debug(f"Traceback : {traceback.format_exc()}")
        return False


def main():
    logger.info("=" * 60)
    logger.info(f"Démarrage de l'extraction avec {MODEL}")
    logger.info(f"Répertoire d'entrée : {INPUT_DIR}")
    logger.info(f"Répertoire de sortie : {OUTPUT_DIR}")
    logger.info(f"URL Ollama : {OLLAMA_BASE_URL}")
    logger.info("=" * 60)

    # Traitement des fichiers PDF
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    pdf_files.sort()
    
    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans {INPUT_DIR}")
        return

    total_files = len(pdf_files)
    logger.info(f"Nombre de fichiers PDF à traiter : {total_files}")

    for file_index, pdf_path in enumerate(pdf_files, start=1):
        pdf_name = os.path.basename(pdf_path)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Fichier {file_index}/{total_files} : {pdf_name}")
        logger.info("=" * 60)
        
        # Reset explicite du contexte LLM pour chaque nouveau fichier
        logger.info(f"🔄 Reset du contexte LLM - Nouveau fichier : {pdf_name}")
        
        # LLM pour la détection (sans format JSON)
        llm_detection = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            temperature=0.1
        )
        
        # LLM pour l'extraction (avec format JSON)
        llm_extraction = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            timeout=TIMEOUT,
            format="json",
            temperature=0.1
        )
        logger.info(f"Instances LLM créées : {MODEL} @ {OLLAMA_BASE_URL}")
        
        # Chargement du PDF (texte ou images pour OCR)
        pdf_text, pdf_images = load_pdf_text_or_images(pdf_path)
        
        if pdf_text is None and pdf_images is None:
            logger.error(f"Échec du chargement du PDF : {pdf_name}")
            continue
        
        # Détection du type de document
        doc_type = detect_document_type(pdf_text, pdf_images, pdf_name, llm_detection)
        
        if doc_type is None:
            logger.warning(f"Impossible de détecter le type de document, utilisation par défaut : devis")
            doc_type = "devis"
        
        # Extraction des données avec le LLM (texte ou OCR via vision)
        structured_data = extract_data_with_llm(pdf_text, pdf_images, pdf_name, doc_type, llm_extraction)
        
        if structured_data is None:
            logger.error(f"Échec de l'extraction des données pour : {pdf_name}")
            continue
        
        # Sauvegarde selon le type de document
        if doc_type == "devis":
            output_name = pdf_name.replace(".pdf", ".json")
            output_path = os.path.join(OUTPUT_DIR, output_name)
            if not save_json(structured_data, output_path):
                logger.error(f"Échec de la sauvegarde JSON pour : {pdf_name}")
                continue
        else:  # facture
            output_name = pdf_name.replace(".pdf", ".csv")
            output_path = os.path.join(OUTPUT_DIR, output_name)
            if not save_csv(structured_data, output_path):
                logger.error(f"Échec de la sauvegarde CSV pour : {pdf_name}")
                continue
        
        logger.info(f"✅ Traitement terminé avec succès pour : {pdf_name} (type: {doc_type})")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✨ Traitement terminé ! {total_files} fichier(s) traité(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
