import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import copy
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- 1. Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_CLIENTS = 3       
ROUNDS = 5            
LOCAL_EPOCHS = 3      
LR = 0.001
DATA_PATH = "./data/auto_vi"
CATEGORY = "pipe_staple"  # Change here for 'pipe_clip' or 'engine_wiring'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Custom Dataset ---
class PipeStapleDataset(Dataset):
    def __init__(self, root_dir, category, transform=None):
        self.transform = transform
        self.samples = [] 
        
        # Handle nested paths
        base_path = os.path.join(root_dir, category, category)
        if not os.path.exists(base_path):
            base_path = os.path.join(root_dir, category)
            
        print(f"--- Loading data from: {base_path} ---")

        for split in ['train', 'test']:
            split_dir = os.path.join(base_path, split)
            if not os.path.exists(split_dir): continue

            for class_name in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_path): continue

                # Label Logic: 0 = OK (Good), 1 = NOK (Anomaly/Missing/etc)
                label = 0 if class_name.lower() == 'good' else 1
                
                image_files = glob.glob(os.path.join(class_path, "*.png"))
                for img_p in image_files:
                    self.samples.append((img_p, label))

        self.targets = [s[1] for s in self.samples]
        if len(self.samples) == 0: raise ValueError("No images found!")
        
        # Stats
        print(f"Total images: {len(self.samples)} (OK: {self.targets.count(0)} / Anomaly: {self.targets.count(1)})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

# --- 3. Simple CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 4. Data Loading & Splitting ---
def load_data():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PipeStapleDataset(DATA_PATH, CATEGORY, transform=transform)
    
    # Global Train/Test Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    global_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Non-IID Client Split (Sorted by label to create bias)
    train_indices = np.array(train_dataset.indices)
    train_targets = np.array(dataset.targets)[train_indices]
    
    sorted_indices = np.argsort(train_targets)
    client_shards = np.array_split(sorted_indices, NUM_CLIENTS)
    
    client_loaders = []
    print("\n--- Client Distribution (Non-IID) ---")
    for i, shard in enumerate(client_shards):
        subset = Subset(dataset, train_indices[shard])
        client_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True))
        print(f"Client {i+1}: {len(subset)} images assigned.")
        
    return client_loaders, global_test_loader

# --- 5. FL Functions & Evaluation ---
def local_train(model, loader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def federated_average(global_w, local_ws):
    avg_w = copy.deepcopy(global_w)
    for key in avg_w.keys():
        avg_w[key] = torch.stack([w[key] for w in local_ws]).mean(dim=0)
    return avg_w

def evaluate(model, loader):
    """Full evaluation with sklearn metrics"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate metrics
    # Target names set to English
    report = classification_report(y_true, y_pred, target_names=['OK', 'Anomaly'], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    accuracy = report['accuracy'] * 100
    f1_anomaly = report['Anomaly']['f1-score']
    
    return accuracy, f1_anomaly, cm, report

# --- 6. Main Execution ---
if __name__ == "__main__":
    print(f"--- 🚀 Starting Federated Learning on {CATEGORY} ---")
    
    try:
        client_loaders, test_loader = load_data()
    except Exception as e:
        print(f"Error: {e}")
        exit()

    global_model = SimpleCNN().to(device)
    global_weights = global_model.state_dict()
    
    # History for plotting
    history_acc = []
    history_f1 = []

    print("\n--- Start of Federated Training ---")
    for r in range(ROUNDS):
        print(f"\n📡 --- Round {r+1}/{ROUNDS} ---")
        local_weights = []
        
        # 1. Local Client Training
        for i in range(NUM_CLIENTS):
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_weights)
            w_client = local_train(client_model, client_loaders[i], epochs=LOCAL_EPOCHS)
            local_weights.append(w_client)
            # print(f"    -> Client {i+1} trained.") 
        
        # 2. Server Aggregation (FedAvg)
        global_weights = federated_average(global_weights, local_weights)
        global_model.load_state_dict(global_weights)
        
        # 3. Global Evaluation
        acc, f1, cm, _ = evaluate(global_model, test_loader)
        history_acc.append(acc)
        history_f1.append(f1)
        
        # Console Display
        print(f" -> Global Accuracy: {acc:.2f}%")
        print(f" -> Anomaly F1-Score: {f1:.2f}")
        
        # Quick Confusion Matrix Analysis
        # cm = [[TN, FP], [FN, TP]] -> [[True OK, False Alarm], [Missed Defect, Caught Defect]]
        if len(cm) == 2:
            missed_defects = cm[1][0]
            print(f" ⚠️  MISSED Defects (False Negatives): {missed_defects}")
        else:
            print(f" Matrix:\n{cm}")

    # --- 7. Final Results & Plotting ---
    print("\n✅ Simulation Completed.")
    
    # Save Model
    model_filename = f"model_{CATEGORY}.pth"
    torch.save(global_model.state_dict(), model_filename)
    print(f"💾 Model saved as: {model_filename}")
    
    # Save Confusion Matrix
    acc, f1, cm, _ = evaluate(global_model, test_loader)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['OK', 'Anomaly'], yticklabels=['OK', 'Anomaly'])
    plt.title(f'Final Confusion Matrix - {CATEGORY}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"confusion_matrix_{CATEGORY}.png")
    print("📊 Confusion matrix saved.")

    # Save Learning Curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ROUNDS + 1), history_acc, 'b-o', label='Global Accuracy')
    plt.plot(range(1, ROUNDS + 1), [x*100 for x in history_f1], 'r--s', label='Anomaly F1-Score')
    plt.title(f'FedAvg Performance - {CATEGORY}')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_{CATEGORY}.png")
    print(f"📈 Charts saved as 'metrics_{CATEGORY}.png'")