from dotenv import load_dotenv
import os, shutil, time
from pathlib import Path
import kagglehub
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

RESULTS = "./results/"

class AI:
    @staticmethod 
    def get_set_loader_ofDataSetImg(transform, path:str, batch:int, shuffle:bool):
        sets = ImageFolder(root=path, transform=transform)
        loader = DataLoader(sets, batch_size=batch, shuffle=shuffle)
        return sets, loader

class SimpleCNN(nn.Module):
    def __init__(self, total_class_num:int):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*10*10, 64),
            nn.ReLU(),
            nn.Linear(64, total_class_num)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class FER2013(AI):
    X, Y = 48, 48
    EMOTION_MAP = {     # Mappa delle emozioni
        0: "Angry", 
        1: "Disgust", 
        2: "Fear", 
        3: "Happy",
        4: "Sad", 
        5: "Surprise", 
        6: "Neutral"
    }

    def __init__(self, model_path:str=None):
        self.model = SimpleCNN(len(self.EMOTION_MAP))
        if model_path is not None:
            self._load_model(model_path)
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((self.X, self.Y)), 
            transforms.ToTensor()
        ])
    
    def _load_model(self, model_path:str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        obj = torch.load(model_path)
        if isinstance(obj, dict):
            self.model.load_state_dict(obj)
            print(f"State dict loaded from {model_path}")
        else:
            self.model = obj
            print(f"Full model loaded from {model_path}")
    
    def save(self, path:str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        print(f"Model saved to {path}")
    
    def save_state_dict(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def download_dataset(path:str):
        load_dotenv()
        try:
            user = kagglehub.whoami()  # Verifica le credenziali
            print(f"✅ Autenticated as: {user}")
        except Exception as e:
            print(f"❌ Autentication Error: {e}")
            raise SystemExit(1) from e
        
        if (path is not None):
            os.makedirs(path, exist_ok=True)
        try:
            cache_path = kagglehub.dataset_download(
                "msambare/fer2013",
                force_download=True
            )
            print(f"🔄 Dataset scaricato nella cache: {cache_path}")
            for item in os.listdir(cache_path):
                src = os.path.join(cache_path, item)
                dst = os.path.join(path, item)
                
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"✅ Dataset finale in: {path}")
            # Pulizia Cache
            user, dataset_name = "msambare/fer2013".split('/')
            cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / user / dataset_name
            shutil.rmtree(cache_root, ignore_errors=True)
            print(f"🗑️  Pulizia cache: {cache_path}")
        except Exception as e:
            print(f"❌ Errore durante l'operazione: {str(e)}")
            if os.path.exists(path):
                shutil.rmtree(path)
            raise SystemExit(1) from e
    
    def EDA(
        self, dataset_path:str,
        batch_train:int=64, batch_test:int=1000,
        shuffle_train:bool=True, shuffle_test:bool=True,
        path_output:str=RESULTS,
        num_sample_img:int=1
    ):
        train_set, train_loader = AI.get_set_loader_ofDataSetImg(self.transform, dataset_path+"/train", batch_train, shuffle_train)
        test_set, _ = AI.get_set_loader_ofDataSetImg(self.transform, dataset_path+"/test", batch_test, shuffle_test)

        print("EDA Reports:")
        print("- Train data size: ", len(train_set))
        print("- Test data size: ", len(test_set))
        print("- Class distribution: ", end="")
        labels = np.array(train_set.targets)
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        for class_idx, count in sorted(class_dist.items()):
            print(f"[{self.EMOTION_MAP[int(class_idx)]}: {count}]", end=" ")
        print()

        os.makedirs(RESULTS, exist_ok=True)
        # Plot class distribution
        plt.figure(figsize=(10,4))
        sns.barplot(x=[self.EMOTION_MAP[k] for k in class_dist.keys()], y=list(class_dist.values()))
        plt.title("Class distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path_output+"class_distribution.png")
        plt.close()
        # Sample Images
        examples = enumerate(train_loader)
        _, (examples_data, examples_targets) = next(examples)
        plt.figure(figsize=(10,4))
        for i in range(min(num_sample_img, len(examples_data))):
            plt.subplot(2,5,i+1)
            plt.imshow(examples_data[i][0], cmap="gray", interpolation="none")
            plt.title(f"Label: {self.EMOTION_MAP[examples_targets[i].item()]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(path_output+"samples_images.png")
        plt.close()
    
    def train(
        self, path_data:str,
        epochs:int=2, batch_size:int=64, shuffle:bool=True
    ):
        _, train_loader = AI.get_set_loader_ofDataSetImg(self.transform, path_data+"/train", batch_size, shuffle)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
        start = time.time()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")
        
        print(f"Training time: {time.time()-start:.2f} seconds")
        print(f"Total Loss: {running_loss/len(train_loader):.4f}")
    
    def evaluate(
        self, path_data:str,
        batch_size:int=1000, shuffle:bool=True, verbose:bool=False,
        path_output:str=RESULTS
    ):
        _, test_loader = AI.get_set_loader_ofDataSetImg(self.transform, path_data+"/test", batch_size, shuffle)
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
        accuracy = correct/total

        if verbose:
            print(f"Test sull'accuratezza: {accuracy:.4f}")
            # Classification Report
            print("Classification Report: ", classification_report(all_labels, all_preds))
            # Plot Confusion Matrix
            name = path_output+"/confusion_matrix.png"
            conf_matrix = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=self.EMOTION_MAP.values(), yticklabels=self.EMOTION_MAP.values())
            plt.xlabel("Emozione Ottenuta")
            plt.ylabel("Emozione Prevista")
            plt.title("Confusion Matrix")
            plt.savefig(name)
            plt.close()
            print(f"Salvata la matrice di confusione in {name}")
    
    def inference_singleImg(self, path_img:str, path_output:str):
        image = read_image(path_img)
        image_tensor = self.transform(to_pil_image(image)).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
            predicted_class = self.EMOTION_MAP[pred]
        
        save_dir = os.path.join(path_output, predicted_class)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(path_img)
        save_path = os.path.join(save_dir, filename)
        save_image(image_tensor.squeeze(0), save_path)
        print(f"Inferenza completata: {predicted_class} -> {save_path}")
