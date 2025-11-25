import os

# Le chemin vers vos données
BASE_PATH = "./data/auto_vi"
# Le nom du fichier de sortie
OUTPUT_FILE = "structure_dossier.txt"

def list_files_to_txt(startpath, output_file):
    print(f"Analyse en cours... Résultat dans '{output_file}'")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- Structure du dossier '{startpath}' ---\n")
        
        if not os.path.exists(startpath):
            f.write(f"ERREUR: Le dossier {startpath} n'existe pas.\n")
            print("Erreur : Dossier introuvable.")
            return

        # On parcourt l'arbre des dossiers
        for root, dirs, files in os.walk(startpath):
            # On calcule le niveau pour l'indentation
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f"{indent}{os.path.basename(root)}/\n")
            
            subindent = ' ' * 4 * (level + 1)
            
            # On note les 5 premiers fichiers pour l'exemple
            for file in files[:5]:
                f.write(f"{subindent}{file}\n")
            
            if len(files) > 5:
                f.write(f"{subindent}... (+{len(files)-5} autres fichiers)\n")

    print("✅ Terminé ! Ouvrez le fichier 'structure_dossier.txt' pour voir le résultat.")

if __name__ == "__main__":
    list_files_to_txt(BASE_PATH, OUTPUT_FILE)