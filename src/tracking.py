"""
Module de tracking des tokens et des statistiques d'utilisation.
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

# Configuration MongoDB
MONGO_HOST = os.getenv("MONGO_HOST", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "documents_db")

_client: Optional[MongoClient] = None
_db = None
_cleanup_done = False  # Flag pour éviter de nettoyer plusieurs fois


def get_connection():
    """Crée ou retourne la connexion MongoDB."""
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
            logger.info(f"✅ Connexion MongoDB établie pour tracking : {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion MongoDB (tracking) : {str(e)}")
            raise
    
    return _db


def track_processing(
    file_name: str,
    doc_type: str,
    supplier: str,
    success: bool,
    duration: float,
    tokens_input: int = 0,
    tokens_output: int = 0,
    char_count: int = 0,
    page_count: int = 0,
    error_message: Optional[str] = None
):
    """
    Enregistre les statistiques de traitement d'un fichier.
    
    Args:
        file_name: Nom du fichier traité
        doc_type: Type de document (facture, devis, bon_livraison)
        supplier: Fournisseur détecté
        success: True si le traitement a réussi
        duration: Durée du traitement en secondes
        tokens_input: Nombre de tokens en entrée
        tokens_output: Nombre de tokens en sortie
        char_count: Nombre de caractères extraits du PDF
        page_count: Nombre de pages du PDF
        error_message: Message d'erreur si échec
    """
    try:
        db = get_connection()
        collection = db["processing_stats"]
        
        stat_entry = {
            "file_name": file_name,
            "doc_type": doc_type,
            "supplier": supplier,
            "success": success,
            "duration": duration,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "char_count": char_count,
            "page_count": page_count,
            "error_message": error_message,
            "timestamp": datetime.utcnow()
        }
        
        # Utiliser upsert pour ne garder qu'une seule entrée par fichier (la dernière tentative)
        # Cela évite de compter plusieurs fois le même fichier en cas de retry
        result = collection.update_one(
            {"file_name": file_name},
            {"$set": stat_entry},
            upsert=True
        )
        
        if result.upserted_id:
            logger.debug(f"✅ Nouvelle statistique créée pour {file_name}")
        elif result.modified_count > 0:
            logger.debug(f"✅ Statistique mise à jour pour {file_name}")
        else:
            logger.debug(f"✅ Statistique déjà à jour pour {file_name}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement des statistiques : {str(e)}")


def cleanup_duplicates():
    """
    Nettoie les doublons existants en gardant seulement la dernière entrée pour chaque fichier.
    Utile pour nettoyer les données avant la mise à jour avec upsert.
    """
    try:
        db = get_connection()
        collection = db["processing_stats"]
        
        # Trouver tous les fichiers en double
        pipeline = [
            {
                "$group": {
                    "_id": "$file_name",
                    "count": {"$sum": 1},
                    "docs": {"$push": {"_id": "$_id", "timestamp": "$timestamp"}}
                }
            },
            {
                "$match": {"count": {"$gt": 1}}
            }
        ]
        
        duplicates = list(collection.aggregate(pipeline))
        
        if duplicates:
            logger.info(f"🧹 Nettoyage de {len(duplicates)} fichiers en double...")
            total_deleted = 0
            for dup in duplicates:
                file_name = dup["_id"]
                docs = dup["docs"]
                # Garder le document le plus récent (dernier timestamp)
                docs_sorted = sorted(
                    docs, 
                    key=lambda x: x.get("timestamp") if x.get("timestamp") else datetime.min, 
                    reverse=True
                )
                latest_id = docs_sorted[0]["_id"]
                
                # Supprimer tous les autres documents pour ce fichier
                ids_to_remove = [doc["_id"] for doc in docs_sorted[1:]]
                if ids_to_remove:
                    result = collection.delete_many({"_id": {"$in": ids_to_remove}})
                    deleted_count = result.deleted_count
                    total_deleted += deleted_count
                    logger.debug(f"   Supprimé {deleted_count} doublon(s) pour {file_name}, gardé le plus récent")
            
            logger.info(f"✅ Nettoyage terminé : {total_deleted} doublon(s) supprimé(s)")
        else:
            logger.debug("✅ Aucun doublon détecté")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des doublons : {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())


def get_dashboard_stats() -> Dict[str, Any]:
    """
    Récupère les statistiques agrégées pour le dashboard.
    
    Returns:
        Dictionnaire contenant toutes les statistiques
    """
    try:
        db = get_connection()
        collection = db["processing_stats"]
        
        # Nettoyer les doublons existants AVANT de compter
        # (le nettoyage est rapide et garantit la cohérence)
        cleanup_duplicates()
        
        # Statistiques globales - compter les fichiers uniques uniquement
        # Chaque fichier n'apparaît qu'une fois (grâce à l'upsert dans track_processing)
        # Utiliser distinct pour être absolument sûr qu'on compte les fichiers uniques
        unique_files = collection.distinct("file_name")
        total_docs = len(unique_files)
        
        # Pour les succès/échecs, on doit compter les fichiers uniques avec leur statut final
        # On prend le dernier statut pour chaque fichier
        pipeline_status = [
            {
                "$sort": {"file_name": 1, "timestamp": -1}  # Trier par nom puis par timestamp décroissant
            },
            {
                "$group": {
                    "_id": "$file_name",
                    "success": {"$first": "$success"},  # Prendre le premier (le plus récent grâce au tri)
                    "timestamp": {"$first": "$timestamp"}
                }
            }
        ]
        
        status_by_file = list(collection.aggregate(pipeline_status))
        successful_docs = sum(1 for item in status_by_file if item.get("success") is True)
        failed_docs = sum(1 for item in status_by_file if item.get("success") is False)
        
        # Agrégation des tokens - utiliser seulement la dernière entrée pour chaque fichier
        # D'abord, garder seulement la dernière entrée par fichier
        pipeline_tokens = [
            {
                "$sort": {"file_name": 1, "timestamp": -1}  # Trier par nom puis par timestamp décroissant
            },
            {
                "$group": {
                    "_id": "$file_name",
                    "tokens_input": {"$first": "$tokens_input"},
                    "tokens_output": {"$first": "$tokens_output"},
                    "char_count": {"$first": "$char_count"},
                    "page_count": {"$first": "$page_count"},
                    "duration": {"$first": "$duration"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_tokens_input": {"$sum": "$tokens_input"},
                    "total_tokens_output": {"$sum": "$tokens_output"},
                    "total_chars": {"$sum": "$char_count"},
                    "total_pages": {"$sum": "$page_count"},
                    "avg_tokens_input": {"$avg": "$tokens_input"},
                    "avg_tokens_output": {"$avg": "$tokens_output"},
                    "avg_chars": {"$avg": "$char_count"},
                    "avg_pages": {"$avg": "$page_count"},
                    "avg_duration": {"$avg": "$duration"}
                }
            }
        ]
        
        tokens_result = list(collection.aggregate(pipeline_tokens))
        tokens_stats = tokens_result[0] if tokens_result else {}
        
        # Statistiques par type de document - utiliser seulement la dernière entrée par fichier
        pipeline_by_type = [
            {
                "$sort": {"file_name": 1, "timestamp": -1}
            },
            {
                "$group": {
                    "_id": {"file_name": "$file_name", "doc_type": "$doc_type"},
                    "tokens_input": {"$first": "$tokens_input"},
                    "tokens_output": {"$first": "$tokens_output"},
                    "duration": {"$first": "$duration"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.doc_type",
                    "count": {"$sum": 1},
                    "tokens_input": {"$sum": "$tokens_input"},
                    "tokens_output": {"$sum": "$tokens_output"},
                    "avg_duration": {"$avg": "$duration"}
                }
            }
        ]
        
        stats_by_type = list(collection.aggregate(pipeline_by_type))
        
        # Statistiques par fournisseur - utiliser seulement la dernière entrée par fichier
        pipeline_by_supplier = [
            {
                "$sort": {"file_name": 1, "timestamp": -1}
            },
            {
                "$group": {
                    "_id": {"file_name": "$file_name", "supplier": "$supplier"},
                    "tokens_input": {"$first": "$tokens_input"},
                    "tokens_output": {"$first": "$tokens_output"},
                    "duration": {"$first": "$duration"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.supplier",
                    "count": {"$sum": 1},
                    "tokens_input": {"$sum": "$tokens_input"},
                    "tokens_output": {"$sum": "$tokens_output"},
                    "avg_duration": {"$avg": "$duration"}
                }
            }
        ]
        
        stats_by_supplier = list(collection.aggregate(pipeline_by_supplier))
        
        # Derniers traitements - seulement la dernière entrée pour chaque fichier
        pipeline_recent = [
            {
                "$sort": {"file_name": 1, "timestamp": -1}
            },
            {
                "$group": {
                    "_id": "$file_name",
                    "doc_type": {"$first": "$doc_type"},
                    "supplier": {"$first": "$supplier"},
                    "success": {"$first": "$success"},
                    "duration": {"$first": "$duration"},
                    "tokens_input": {"$first": "$tokens_input"},
                    "tokens_output": {"$first": "$tokens_output"},
                    "timestamp": {"$first": "$timestamp"}
                }
            },
            {
                "$sort": {"timestamp": -1}
            },
            {
                "$limit": 10
            },
            {
                "$project": {
                    "_id": 0,
                    "file_name": "$_id",
                    "doc_type": 1,
                    "supplier": 1,
                    "success": 1,
                    "duration": 1,
                    "tokens_input": 1,
                    "tokens_output": 1,
                    "timestamp": 1
                }
            }
        ]
        
        recent_processing = list(collection.aggregate(pipeline_recent))
        
        # Calcul des coûts (basé sur les tarifs OpenAI GPT-5 nano)
        # Tarifs officiels GPT-5 nano :
        # - Input: $0.05 par 1M tokens = $0.00005 par 1K tokens
        # - Output: $0.40 par 1M tokens = $0.0004 par 1K tokens
        # - Cached Input: $0.01 par 1M tokens (non utilisé actuellement)
        COST_PER_1K_INPUT = 0.00005  # USD par 1K tokens input (GPT-5 nano)
        COST_PER_1K_OUTPUT = 0.0004  # USD par 1K tokens output (GPT-5 nano)
        
        total_tokens_input = tokens_stats.get("total_tokens_input", 0)
        total_tokens_output = tokens_stats.get("total_tokens_output", 0)
        
        cost_input = (total_tokens_input / 1000) * COST_PER_1K_INPUT
        cost_output = (total_tokens_output / 1000) * COST_PER_1K_OUTPUT
        total_cost = cost_input + cost_output
        
        return {
            "global": {
                "total_documents": total_docs,
                "successful": successful_docs,
                "failed": failed_docs,
                "success_rate": (successful_docs / total_docs * 100) if total_docs > 0 else 0
            },
            "tokens": {
                "total_input": int(tokens_stats.get("total_tokens_input", 0)),
                "total_output": int(tokens_stats.get("total_tokens_output", 0)),
                "total": int(tokens_stats.get("total_tokens_input", 0) + tokens_stats.get("total_tokens_output", 0)),
                "avg_input": round(tokens_stats.get("avg_tokens_input", 0), 2),
                "avg_output": round(tokens_stats.get("avg_tokens_output", 0), 2)
            },
            "content": {
                "total_chars": int(tokens_stats.get("total_chars", 0)),
                "total_pages": int(tokens_stats.get("total_pages", 0)),
                "avg_chars": round(tokens_stats.get("avg_chars", 0), 2),
                "avg_pages": round(tokens_stats.get("avg_pages", 0), 2)
            },
            "performance": {
                "avg_duration": round(tokens_stats.get("avg_duration", 0), 2)
            },
            "costs": {
                "input": round(cost_input, 4),
                "output": round(cost_output, 4),
                "total": round(total_cost, 4),
                "cost_per_1k_input": COST_PER_1K_INPUT,
                "cost_per_1k_output": COST_PER_1K_OUTPUT
            },
            "by_type": [
                {
                    "doc_type": item["_id"],
                    "count": item["count"],
                    "tokens_input": item["tokens_input"],
                    "tokens_output": item["tokens_output"],
                    "avg_duration": round(item["avg_duration"], 2)
                }
                for item in stats_by_type
            ],
            "by_supplier": [
                {
                    "supplier": item["_id"],
                    "count": item["count"],
                    "tokens_input": item["tokens_input"],
                    "tokens_output": item["tokens_output"],
                    "avg_duration": round(item["avg_duration"], 2)
                }
                for item in stats_by_supplier
            ],
            "recent": recent_processing
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques : {str(e)}")
        return {
            "global": {"total_documents": 0, "successful": 0, "failed": 0, "success_rate": 0},
            "tokens": {"total_input": 0, "total_output": 0, "total": 0, "avg_input": 0, "avg_output": 0},
            "content": {"total_chars": 0, "total_pages": 0, "avg_chars": 0, "avg_pages": 0},
            "performance": {"avg_duration": 0},
            "costs": {"input": 0, "output": 0, "total": 0, "cost_per_1k_input": 0, "cost_per_1k_output": 0},
            "by_type": [],
            "by_supplier": [],
            "recent": []
        }
