import os
import requests
import zipfile
from tqdm import tqdm  # Pour la barre de progression

# 1. Configuration des URLs extraites du fichier croissant.json
# Vous pouvez commenter certaines lignes pour ne télécharger qu'une partie au début
DATASET_URLS = {
    "pipe_staple": "https://zenodo.org/records/10459003/files/pipe_staple.zip",   # ~106 MB (Le plus léger pour tester)
    "pipe_clip": "https://zenodo.org/records/10459003/files/pipe_clip.zip",       # ~144 MB
    "engine_wiring": "https://zenodo.org/records/10459003/files/engine_wiring.zip", # ~250 MB
    # "underbody_pipes": "https://zenodo.org/records/10459003/files/underbody_pipes.zip", # ~950 MB
    # "tank_screw": "https://zenodo.org/records/10459003/files/tank_screw.zip",       # ~1.1 GB
    # "underbody_screw": "https://zenodo.org/records/10459003/files/underbody_screw.zip"  # ~1.3 GB
}

BASE_DIR = "./data/auto_vi"

def download_and_extract(name, url, output_dir):
    """Télécharge et décompresse un fichier zip."""
    
    # Créer le dossier cible s'il n'existe pas
    target_path = os.path.join(output_dir, name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    print(f"--- Traitement de {name} ---")
    
    # Nom du fichier zip local
    zip_path = os.path.join(output_dir, f"{name}.zip")
    
    # 1. Téléchargement
    if not os.path.exists(zip_path):
        print(f"Téléchargement depuis {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc=name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print("Fichier zip déjà présent.")

    # 2. Extraction
    print(f"Extraction dans {target_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        print("Extraction terminée.")
    except zipfile.BadZipFile:
        print("Erreur : Le fichier zip semble corrompu.")

    # Optionnel : Supprimer le zip pour économiser de l'espace
    # os.remove(zip_path) 

# Exécution du script
if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    
    print(f"Préparation du téléchargement dans {BASE_DIR}...")
    for name, url in DATASET_URLS.items():
        download_and_extract(name, url, BASE_DIR)
    
    print("\n✅ Tout est prêt ! Vos données sont dans le dossier 'data/auto_vi'.")