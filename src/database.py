import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)

# Configuration MongoDB depuis les variables d'environnement
MONGO_HOST = os.getenv("MONGO_HOST", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "documents_db")

# Client MongoDB global (singleton)
_client: Optional[MongoClient] = None
_db = None


def get_connection():
    """
    Crée ou retourne la connexion MongoDB.
    """
    global _client, _db
    
    if _client is None:
        try:
            _client = MongoClient(
                host=MONGO_HOST,
                port=MONGO_PORT,
                serverSelectionTimeoutMS=5000
            )
            _client.admin.command('ping')
            _db = _client[MONGO_DB_NAME]
            logger.info(f"✅ Connexion MongoDB établie : {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion MongoDB : {str(e)}")
            raise
    
    return _db


def insert_document(supplier: str, doc_type: str, data: Dict[str, Any], contenu_fichier: Optional[str] = None, type_fichier: Optional[str] = None):
    try:
        db = get_connection()
        
        # 1. On définit les 3 collections
        collection_map = {
            "facture": "factures",
            "devis": "devis",
            "bon_livraison": "bons_livraison"
        }
        collection_name = collection_map.get(doc_type, "autres")
        collection = db[collection_name]
        
        fichier_source = data.get("fichier_source", "inconnu")
        supplier_id = supplier.lower().strip().replace(" ", "_")

        # 2. On prépare la structure "Document par Fichier"
        # Cette structure est plate et facile à requêter
        doc_entry = {
            "metadata": {
                "fournisseur_id": supplier_id,
                "nom_fournisseur": supplier,
                "fichier_source": fichier_source,
                "date_extraction": datetime.utcnow(),
                "format_export": type_fichier
            },
            # On sépare les données pour la clarté
            "entete": data.get("entete"),
            "lignes": data.get("lignes") if doc_type != "devis" else None,
            "prestations": data.get("prestations") if doc_type == "devis" else None,
            "totaux": data.get("totaux") if doc_type == "devis" else None,
            "contenu_export_brut": contenu_fichier # Ton CSV ou JSON textuel
        }

        # 3. MISE À JOUR (Upsert) 
        # On cherche par nom de fichier. Si le fichier existe, on le met à jour.
        # Sinon, on en crée un nouveau.
        collection.update_one(
            {"metadata.fichier_source": fichier_source},
            {"$set": doc_entry},
            upsert=True
        )
        
        logger.info(f"✅ MongoDB : Document '{fichier_source}' enregistré dans '{collection_name}' (Fournisseur: {supplier_id})")

    except Exception as e:
        logger.error(f"❌ Erreur MongoDB : {str(e)}")
        raise