import os
import shutil
import logging
import time
import uuid
import asyncio
import pandas as pd
import io
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bson import ObjectId
from datetime import datetime, timedelta
from src.database import (
    get_connection, 
    get_all_suppliers, 
    get_documents_by_supplier
)
from src.main import build_graph, process_single_file, register_file_in_progress, unregister_file_in_progress

logger = logging.getLogger(__name__)

# Chemins
INPUT_DIR = "/app/input"
TEMPLATES_DIR = "/app/src/templates"
STATIC_DIR = "/app/src/static"

# Créer les dossiers si nécessaire
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Application FastAPI
app = FastAPI(title="Gestion Documents Comptables")

# Templates et fichiers statiques
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Graphe LangGraph (partagé)
_langgraph_app = None

# Pool d'exécuteurs pour le traitement parallèle (max 3 documents)
executor = ThreadPoolExecutor(max_workers=3)

# Dictionnaire pour suivre l'état des tâches
# Format : {task_id: {status: str, file_name: str, message: str, success: bool, timestamp: datetime}}
task_status = {}

# Cache des stats (TTL 30s)
_stats_cache: dict = {"data": None, "ts": 0}
STATS_CACHE_TTL = 30


def invalidate_stats_cache() -> None:
    """Invalide le cache des stats (après insertion d'un document)."""
    global _stats_cache
    _stats_cache["data"] = None
    _stats_cache["ts"] = 0


def get_langgraph_app():
    """Récupère ou crée l'instance du graphe LangGraph."""
    global _langgraph_app
    if _langgraph_app is None:
        _langgraph_app = build_graph()
    return _langgraph_app


def convert_objectid(obj):
    """Convertit ObjectId en string pour JSON."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def get_collection_stats():
    """Récupère les statistiques de toutes les collections (cache TTL 30s)."""
    global _stats_cache
    now = time.time()
    if _stats_cache["data"] is not None and (now - _stats_cache["ts"]) < STATS_CACHE_TTL:
        return _stats_cache["data"]
    db = get_connection()
    collections = {
        "factures": db["factures"],
        "devis": db["devis"],
        "bons_livraison": db["bons_livraison"]
    }
    stats = {}
    for name, collection in collections.items():
        count = collection.count_documents({})
        stats[name] = {
            "count": count,
            "display_name": name.replace("_", " ").title()
        }
    _stats_cache["data"] = stats
    _stats_cache["ts"] = now
    return stats


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page d'accueil avec vue d'ensemble."""
    stats = get_collection_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats
    })


@app.get("/collections", response_class=HTMLResponse)
async def collections_page(request: Request):
    """Page listant toutes les collections."""
    stats = get_collection_stats()
    return templates.TemplateResponse("collections.html", {
        "request": request,
        "stats": stats
    })


@app.get("/collections/{collection_name}", response_class=HTMLResponse)
async def collection_documents(
    request: Request, 
    collection_name: str, 
    page: int = 1, 
    limit: int = 20,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fournisseur: Optional[str] = None,
    client: Optional[str] = None
):
    """Liste des documents d'une collection avec pagination et filtres."""
    db = get_connection()
    
    # Vérifier que la collection existe
    valid_collections = ["factures", "devis", "bons_livraison"]
    if collection_name not in valid_collections:
        raise HTTPException(status_code=404, detail="Collection non trouvée")
    
    collection = db[collection_name]
    
    # Construire la requête de filtrage
    query = {}
    
    if search:
        # Recherche dans plusieurs champs
        query["$or"] = [
            {"metadata.fichier_source": {"$regex": search, "$options": "i"}},
            {"metadata.nom_fournisseur": {"$regex": search, "$options": "i"}},
            {"entete.numero_facture": {"$regex": search, "$options": "i"}},
            {"entete.numero_devis": {"$regex": search, "$options": "i"}},
            {"entete.client_nom": {"$regex": search, "$options": "i"}}
        ]
    
    if fournisseur:
        query["metadata.nom_fournisseur"] = fournisseur
    
    if client:
        query["entete.client_nom"] = client
    
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = datetime.fromisoformat(date_from)
        if date_to:
            # Ajouter la fin de journée
            date_to_end = datetime.fromisoformat(date_to) + timedelta(days=1) - timedelta(seconds=1)
            date_query["$lte"] = date_to_end
        
        # Déterminer le champ de date selon le type de collection
        if collection_name == "factures":
            query["entete.date"] = date_query
        elif collection_name == "devis":
            query["entete.date_emission"] = date_query
        else:
            query["metadata.date_extraction"] = date_query
    
    # Pagination
    skip = (page - 1) * limit
    total = collection.count_documents(query)
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    
    # Récupérer les documents
    documents = list(collection.find(query).sort("metadata.date_extraction", -1).skip(skip).limit(limit))
    
    # Convertir ObjectId et datetime
    documents = [convert_objectid(doc) for doc in documents]
    
    return templates.TemplateResponse("collection_documents.html", {
        "request": request,
        "collection_name": collection_name,
        "display_name": collection_name.replace("_", " ").title(),
        "documents": documents,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages
    })


@app.get("/documents/{collection_name}/{document_id}", response_class=HTMLResponse)
async def document_detail(request: Request, collection_name: str, document_id: str):
    """Détails d'un document spécifique."""
    db = get_connection()
    
    valid_collections = ["factures", "devis", "bons_livraison"]
    if collection_name not in valid_collections:
        raise HTTPException(status_code=404, detail="Collection non trouvée")
    
    collection = db[collection_name]
    
    try:
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            raise HTTPException(status_code=404, detail="Document non trouvé")
        
        document = convert_objectid(document)
        
        return templates.TemplateResponse("document.html", {
            "request": request,
            "collection_name": collection_name,
            "display_name": collection_name.replace("_", " ").title(),
            "document": document
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du document : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{collection_name}/{document_id}/excel")
async def export_document_excel(collection_name: str, document_id: str):
    """Exporte un document spécifique au format Excel."""
    db = get_connection()
    valid_collections = ["factures", "devis", "bons_livraison"]
    if collection_name not in valid_collections:
        raise HTTPException(status_code=404, detail="Collection non trouvée")
    
    collection = db[collection_name]
    
    try:
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            raise HTTPException(status_code=404, detail="Document non trouvé")
        
        # Préparation des données pour Excel
        entete = document.get("entete", {})
        metadata = document.get("metadata", {})
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Onglet Entête
            df_entete = pd.DataFrame([entete])
            df_entete.to_excel(writer, sheet_name='Entete', index=False)
            
            # Onglet Lignes/Prestations
            if collection_name == "devis":
                lignes = document.get("prestations", [])
                sheet_name = 'Prestations'
            else:
                lignes = document.get("lignes", [])
                sheet_name = 'Lignes'
            
            if lignes:
                df_lignes = pd.DataFrame(lignes)
                # Ajouter les infos d'entête à chaque ligne pour faciliter le traitement Excel
                for key, value in entete.items():
                    if key not in df_lignes.columns:
                        df_lignes[key] = value
                df_lignes.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Onglet Metadata
            df_meta = pd.DataFrame([metadata])
            df_meta.to_excel(writer, sheet_name='Metadata', index=False)
        
        output.seek(0)
        filename = metadata.get("fichier_source", "document").replace(".pdf", ".xlsx")
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export Excel : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suppliers", response_class=HTMLResponse)
async def suppliers_page(request: Request):
    """Page listant tous les fournisseurs."""
    suppliers = get_all_suppliers()
    return templates.TemplateResponse("suppliers.html", {
        "request": request,
        "suppliers": suppliers
    })


@app.get("/suppliers/{supplier_name}", response_class=HTMLResponse)
async def supplier_detail(request: Request, supplier_name: str):
    """Détails d'un fournisseur avec agrégation de tous ses documents."""
    docs = get_documents_by_supplier(supplier_name)
    
    # Convertir les ObjectIds pour les templates
    for doc_type in docs:
        docs[doc_type] = [convert_objectid(doc) for doc in docs[doc_type]]
    
    return templates.TemplateResponse("supplier_detail.html", {
        "request": request,
        "supplier_name": supplier_name,
        "documents": docs
    })


@app.get("/suppliers/{supplier_name}/aggregated", response_class=HTMLResponse)
async def supplier_aggregated(request: Request, supplier_name: str):
    """Tableaux agrégés de tous les documents d'un fournisseur."""
    docs = get_documents_by_supplier(supplier_name)
    
    # Convertir les ObjectIds pour les templates
    for doc_type in docs:
        docs[doc_type] = [convert_objectid(doc) for doc in docs[doc_type]]
    
    # Convertir le nom du fournisseur en ID (minuscules, sans espaces)
    supplier_id = supplier_name.lower().strip().replace(" ", "_")
    
    return templates.TemplateResponse("supplier_aggregated.html", {
        "request": request,
        "supplier_name": supplier_name,
        "supplier_id": supplier_id,
        "documents": docs
    })


def run_processing_task(task_id: str, file_path: str, filename: str):
    """Exécute le traitement d'un fichier dans le pool d'exécuteurs."""
    global task_status
    
    try:
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["message"] = f"Traitement de {filename} en cours..."
        
        app_langgraph = get_langgraph_app()
        
        # Traiter le fichier avec LangGraph
        # process_single_file est synchrone, on l'appelle dans le pool
        pdf_name_result, success, elapsed, error_msg = process_single_file(
            app_langgraph, 
            file_path, 
            1, 
            1
        )
        
        if success:
            task_status[task_id]["status"] = "completed"
            task_status[task_id]["success"] = True
            task_status[task_id]["message"] = f"Fichier {filename} traité avec succès en {elapsed:.1f}s"
            invalidate_stats_cache()
            # Supprimer le PDF de input après traitement réussi
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"🗑️ Fichier supprimé après traitement : {file_path}")
                except Exception as e:
                    logger.error(f"⚠️ Erreur lors de la suppression de {file_path} : {str(e)}")
        else:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["success"] = False
            task_status[task_id]["message"] = f"Erreur : {error_msg or 'Inconnue'}"
            # On garde le fichier en cas d'erreur pour analyse ? 
            # L'utilisateur a dit "supprime les pdf de input une fois traités". 
            # Généralement "traité" implique un succès. Je vais le garder en cas d'échec pour l'instant.
            
    except Exception as e:
        logger.error(f"❌ Erreur critique tâche {task_id} ({filename}) : {str(e)}")
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["success"] = False
        task_status[task_id]["message"] = f"Erreur critique : {str(e)}"
    finally:
        unregister_file_in_progress(file_path)

@app.post("/upload")
async def upload_pdf(request: Request, background_tasks: BackgroundTasks):
    """Upload et lancement du traitement asynchrone."""
    try:
        form = await request.form()
        uploaded_files = form.getlist("files")
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="Aucun fichier fourni")
        
        pdf_files = []
        for f in uploaded_files:
            if hasattr(f, 'filename') and f.filename:
                if f.filename.lower().endswith('.pdf'):
                    pdf_files.append(f)
        
        if not pdf_files:
            raise HTTPException(status_code=400, detail="Aucun fichier PDF valide")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des fichiers : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de la récupération des fichiers : {str(e)}")
    
    response_tasks = []
    
    for file in pdf_files:
        task_id = str(uuid.uuid4())
        file_path = os.path.join(INPUT_DIR, file.filename)
        
        # Marquer comme en cours AVANT de sauvegarder (évite que le watcher le traite en parallèle)
        register_file_in_progress(file_path)
        
        # Sauvegarder le fichier immédiatement
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialiser le statut
        task_status[task_id] = {
            "status": "pending",
            "file_name": file.filename,
            "message": "En attente de traitement...",
            "success": None,
            "timestamp": datetime.now()
        }
        
        # Lancer la tâche dans le pool via BackgroundTasks pour ne pas bloquer
        background_tasks.add_task(executor.submit, run_processing_task, task_id, file_path, file.filename)
        
        response_tasks.append({
            "task_id": task_id,
            "file_name": file.filename
        })
    
    return JSONResponse({
        "success": True,
        "message": f"{len(response_tasks)} fichier(s) mis en file d'attente",
        "tasks": response_tasks
    })

@app.get("/api/upload/status")
async def get_upload_status():
    """Récupère l'état des tâches d'upload récentes."""
    # Nettoyer les vieilles tâches (plus de 1 heure)
    now = datetime.now()
    to_delete = [tid for tid, info in task_status.items() 
                 if now - info["timestamp"] > timedelta(hours=1)]
    for tid in to_delete:
        del task_status[tid]
        
    # Retourner les statuts triés par timestamp décroissant
    sorted_tasks = sorted(
        [{"id": tid, **info} for tid, info in task_status.items()],
        key=lambda x: x["timestamp"],
        reverse=True
    )
    
    # Convertir les datetime en string pour le JSON
    for task in sorted_tasks:
        if isinstance(task["timestamp"], datetime):
            task["timestamp"] = task["timestamp"].isoformat()
            
    return JSONResponse(sorted_tasks)


@app.get("/api/collections")
async def api_collections():
    """API JSON pour les collections."""
    stats = get_collection_stats()
    return JSONResponse(stats)


@app.get("/api/collections/{collection_name}")
async def api_collection_documents(
    collection_name: str, 
    page: int = 1, 
    limit: int = 20,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fournisseur: Optional[str] = None,
    client: Optional[str] = None
):
    """API JSON pour les documents d'une collection avec filtres."""
    db = get_connection()
    
    valid_collections = ["factures", "devis", "bons_livraison"]
    if collection_name not in valid_collections:
        raise HTTPException(status_code=404, detail="Collection non trouvée")
    
    collection = db[collection_name]
    
    # Construire la requête de filtrage (même logique que la route HTML)
    query = {}
    
    if search:
        query["$or"] = [
            {"metadata.fichier_source": {"$regex": search, "$options": "i"}},
            {"metadata.nom_fournisseur": {"$regex": search, "$options": "i"}},
            {"entete.numero_facture": {"$regex": search, "$options": "i"}},
            {"entete.numero_devis": {"$regex": search, "$options": "i"}},
            {"entete.client_nom": {"$regex": search, "$options": "i"}}
        ]
    
    if fournisseur:
        query["metadata.nom_fournisseur"] = fournisseur
    
    if client:
        query["entete.client_nom"] = client
    
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = datetime.fromisoformat(date_from)
        if date_to:
            date_to_end = datetime.fromisoformat(date_to) + timedelta(days=1) - timedelta(seconds=1)
            date_query["$lte"] = date_to_end
        
        if collection_name == "factures":
            query["entete.date"] = date_query
        elif collection_name == "devis":
            query["entete.date_emission"] = date_query
        else:
            query["metadata.date_extraction"] = date_query
    
    skip = (page - 1) * limit
    total = collection.count_documents(query)
    
    documents = list(collection.find(query).sort("metadata.date_extraction", -1).skip(skip).limit(limit))
    documents = [convert_objectid(doc) for doc in documents]
    
    return JSONResponse({
        "collection": collection_name,
        "page": page,
        "limit": limit,
        "total": total,
        "documents": documents
    })
