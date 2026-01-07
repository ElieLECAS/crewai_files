import os
import json
from dotenv import load_dotenv
from crew import InvoiceProcessingCrew

def main():
    load_dotenv()
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    for filename in files:
        print(f"\n🔄 Traitement isolé de {filename}...")
        # Création d'une instance fraîche pour chaque PDF
        processing_crew = InvoiceProcessingCrew() 
        pdf_path = f"/app/input/{filename}"
        
        try:
            result = processing_crew.kickoff(
                pdf_path=pdf_path, 
                pdf_filename=filename
            )
            
            # Logique de sauvegarde (réutilisez celle de votre main.py initial)
            output_filename = filename.replace('.pdf', '.json')
            output_path = os.path.join(output_dir, output_filename)
            
            # Conversion du résultat Pydantic en dictionnaire pour JSON
            data = result.model_dump() if hasattr(result, 'model_dump') else result
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Terminé : {output_filename}")
            
        except Exception as e:
            print(f"❌ Échec sur {filename}: {e}")

if __name__ == "__main__":
    main()