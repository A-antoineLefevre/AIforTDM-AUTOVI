import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

# 1. Configuration des Hyperparamètres
IMG_SIZE = 128        # Taille réduite pour aller vite (128x128 ou 256x256)
BATCH_SIZE = 32       # Nombre d'images traitées en une fois
DATA_PATH = "./data/auto_vi" # Le dossier où vous avez téléchargé les zips

def get_dataloaders(data_dir, category="pipe_staple"):
    """
    Crée les dataloaders Train et Test pour une catégorie spécifique.
    """
    target_dir = os.path.join(data_dir, category)
    
    # Vérification que le dossier existe
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Le dossier {target_dir} n'existe pas. Avez-vous lancé le téléchargement ?")

    # 2. Pipeline de transformation (Prétraitement)
    # - Resize : Uniformise la taille
    # - ToTensor : Convertit l'image en chiffres (0 à 1)
    # - Normalize : Centre les données (standard pour les réseaux de neurones)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Normalisation standard (moyenne, écart-type)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # 3. Chargement des images depuis le dossier
    # ImageFolder suppose une structure dossier/classe/image. 
    # Si vos images sont en vrac, nous utilisons un dossier racine temporaire ou une astuce.
    # Ici, on pointe vers le dossier contenant les sous-dossiers d'images.
    try:
        full_dataset = datasets.ImageFolder(root=target_dir, transform=transform)
    except:
        # Fallback si la structure n'est pas standard "Dossier/Classe"
        # On pointe sur le parent pour qu'il prenne le dossier "pipe_staple" comme une classe
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        # On filtre pour ne garder que ceux de la catégorie visée si besoin (avancé)
    
    print(f"--- Catégorie : {category} ---")
    print(f"Nombre total d'images trouvées : {len(full_dataset)}")

    # 4. Séparation Train / Test (80% / 20%)
    # Dans un cas réel industriel, on s'assure que le Train ne contient QUE des pièces saines.
    # Ici, on fait un split aléatoire pour la Baseline.
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Images d'entraînement : {len(train_dataset)}")
    print(f"Images de test : {len(test_dataset)}")

    # 5. Création des chargeurs (Loaders)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# --- Test du code ---
if __name__ == "__main__":
    # Essayons avec "pipe_staple" (ou changez par "engine_wiring", etc.)
    try:
        train_loader, test_loader = get_dataloaders(DATA_PATH, category="pipe_staple")
        
        # Vérification : on récupère un batch pour voir les dimensions
        images, labels = next(iter(train_loader))
        print(f"Dimensions d'un batch : {images.shape}") 
        # Doit afficher : torch.Size([32, 3, 128, 128]) -> [Batch, Channels, Height, Width]
        
    except Exception as e:
        print(f"Erreur : {e}")