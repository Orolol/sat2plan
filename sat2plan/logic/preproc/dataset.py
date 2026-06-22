import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class Satellite2Map_Data(Dataset):
    def __init__(self, root, image_size=256, verbose=False):
        self.root = root
        self.image_size = image_size
        self.verbose = verbose
        
        # Filtrer uniquement les fichiers existants et valides
        all_files = os.listdir(self.root)
        self.list_files = []
        
        if self.verbose:
            print(f"Scanning {len(all_files)} files in {root}...")
        for file in all_files:
            file_path = os.path.join(self.root, file)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    if self.verbose:
                        print(f"Checking file: {file_path}")
                    # Test rapide d'ouverture pour vérifier l'intégrité
                    # with Image.open(file_path) as img:
                    #     img.verify()  # Vérification de l'intégrité
                    self.list_files.append(file)
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping corrupted file {file}: {e}")
                    # Supprimer le fichier corrompu
                    try:
                        os.remove(file_path)
                        if self.verbose:
                            print(f"Deleted corrupted file: {file_path}")
                    except:
                        pass
            else:
                if self.verbose:
                    print(f"Skipping missing or empty file: {file}")
        
        if self.verbose:
            print(f"Found {len(self.list_files)} valid files out of {len(all_files)}")
        
        # Transformations de base (redimensionnement et normalisation)
        self.resize_transform = transforms.Resize((image_size, image_size), antialias=True)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Transformations spécifiques à l'image satellite
        self.satellite_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3)
        ])

    def apply_joint_transforms(self, satellite_img, map_img):
        # Appliquer exactement les mêmes transformations aux deux images
        if random.random() < 0.5:
            satellite_img = transforms.functional.hflip(satellite_img)
            map_img = transforms.functional.hflip(map_img)
            
        if random.random() < 0.5:
            satellite_img = transforms.functional.vflip(satellite_img)
            map_img = transforms.functional.vflip(map_img)
            
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            satellite_img = transforms.functional.rotate(satellite_img, angle)
            map_img = transforms.functional.rotate(map_img, angle)
            
        return satellite_img, map_img

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        # Gestion robuste des erreurs avec fallback
        max_retries = 5
        for attempt in range(max_retries):
            try:
                current_index = (index + attempt) % len(self.list_files)
                img_file = self.list_files[current_index]
                img_path = os.path.join(self.root, img_file)
                
                # Vérifier l'existence du fichier
                if not os.path.exists(img_path):
                    print(f"File missing: {img_path}")
                    continue
                
                # Charger l'image avec PIL
                image = Image.open(img_path)
                break
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                if attempt == max_retries - 1:
                    # Dernière tentative échouée, créer une image dummy
                    print(f"Creating dummy image after {max_retries} failed attempts")
                    image = Image.new('RGB', (512, 256), color='black')
                    break
                continue
        
        # Séparer l'image en deux (satellite et plan)
        w = image.width
        satellite_img = image.crop((0, 0, w//2, image.height))
        map_img = image.crop((w//2, 0, w, image.height))
        
        # Redimensionner les images
        satellite_img = self.resize_transform(satellite_img)
        map_img = self.resize_transform(map_img)
        
        # Appliquer les mêmes transformations aux deux images
        satellite_img, map_img = self.apply_joint_transforms(satellite_img, map_img)
        
        # Appliquer les transformations spécifiques à l'image satellite
        satellite_img = self.satellite_transform(satellite_img)
        
        # Convertir en tenseurs et normaliser
        satellite_img = self.normalize(self.to_tensor(satellite_img))
        map_img = self.normalize(self.to_tensor(map_img))
        
        return satellite_img, map_img, False
