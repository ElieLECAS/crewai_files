from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import json
import os


class JsonFileWriterInput(BaseModel):
    """Schéma d'entrée pour l'outil d'écriture de fichiers JSON."""
    filename: str = Field(..., description="Le nom du fichier à créer (ex: devis_001.json)")
    content: Any = Field(
        ..., 
        description="Le contenu à écrire. Peut être une chaîne JSON, un dictionnaire Python, une liste, "
                   "ou tout autre objet sérialisable en JSON. Si c'est un objet Python (dict/list), "
                   "il sera automatiquement converti en chaîne JSON formatée."
    )
    directory: str = Field(default="./", description="Le répertoire où sauvegarder le fichier (relatif au répertoire de sortie)")
    overwrite: bool = Field(default=True, description="Si True, écrase le fichier s'il existe déjà")


class JsonFileWriterTool(BaseTool):
    name: str = "json_file_writer"
    description: str = (
        "Outil OBLIGATOIRE pour sauvegarder les données extraites dans un fichier JSON. "
        "Vous DEVEZ utiliser cet outil pour créer le fichier de sortie. "
        "Accepte un objet JSON (dict/list) ou une chaîne JSON. "
        "Si un objet Python (dict/list) est fourni, il sera automatiquement converti en chaîne JSON formatée. "
        "Utilisez cet outil après avoir extrait les données du PDF pour sauvegarder le résultat. "
        "Exemple: filename='devis_001.json', content={'numero_devis': 'DEV-0001', ...}, directory='./'"
    )
    args_schema: Type[BaseModel] = JsonFileWriterInput

    def __init__(self, root_dir: str = "./", **kwargs):
        super().__init__(**kwargs)
        # Utiliser object.__setattr__ pour contourner la validation Pydantic
        object.__setattr__(self, '_root_dir', root_dir)

    def _run(
        self, 
        filename: str, 
        content: Any, 
        directory: str = "./",
        overwrite: bool = True
    ) -> str:
        """
        Écrit un fichier JSON avec le contenu fourni.
        
        Args:
            filename: Le nom du fichier à créer
            content: Le contenu à écrire (chaîne JSON, dict ou list)
            directory: Le répertoire où sauvegarder (relatif à root_dir)
            overwrite: Si True, écrase le fichier s'il existe
        
        Returns:
            Un message de confirmation avec le chemin du fichier créé
        """
        try:
            # Construire le chemin complet
            if directory.startswith("/"):
                # Chemin absolu
                full_dir = directory
            else:
                # Chemin relatif à root_dir - normaliser pour éviter ./ dans le chemin
                if directory == "./" or directory == ".":
                    full_dir = self._root_dir
                else:
                    full_dir = os.path.join(self._root_dir, directory)
            
            # Normaliser le chemin pour éviter les ./ et autres éléments redondants
            full_dir = os.path.normpath(full_dir)
            os.makedirs(full_dir, exist_ok=True)
            
            file_path = os.path.join(full_dir, filename)
            # Normaliser aussi le chemin final
            file_path = os.path.normpath(file_path)
            
            # Vérifier si le fichier existe
            if os.path.exists(file_path) and not overwrite:
                return f"Erreur: Le fichier {file_path} existe déjà et overwrite=False"
            
            # Convertir le contenu en chaîne JSON si nécessaire
            if isinstance(content, (dict, list)):
                # C'est un objet Python, le convertir en JSON formaté
                json_string = json.dumps(content, ensure_ascii=False, indent=2)
            elif isinstance(content, str):
                # C'est déjà une chaîne, vérifier si c'est du JSON valide
                try:
                    # Essayer de parser pour valider
                    parsed = json.loads(content)
                    # Re-formater pour avoir un JSON propre
                    json_string = json.dumps(parsed, ensure_ascii=False, indent=2)
                except (json.JSONDecodeError, TypeError):
                    # Si ce n'est pas du JSON valide, on l'écrit tel quel
                    json_string = content
            else:
                # Type non supporté, essayer de convertir en JSON
                json_string = json.dumps(content, ensure_ascii=False, indent=2)
            
            # Écrire le fichier
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            
            return f"Fichier JSON créé avec succès : {file_path}"
        
        except Exception as e:
            return f"Erreur lors de l'écriture du fichier {filename}: {str(e)}"


# Fonction helper pour créer une instance avec root_dir
def create_json_file_writer(root_dir: str = "./"):
    """Crée une instance de JsonFileWriterTool avec le répertoire racine spécifié."""
    tool = JsonFileWriterTool(root_dir=root_dir)
    return tool

