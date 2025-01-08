import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class Satellite2Map_Data(Dataset):
    def __init__(self, root, image_size=256):
        self.root = root
        self.image_size = image_size
        self.list_files = os.listdir(self.root)
        
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
        img_file = self.list_files[index]
        img_path = os.path.join(self.root, img_file)
        
        # Charger l'image avec PIL
        image = Image.open(img_path)
        
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
