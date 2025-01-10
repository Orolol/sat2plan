from google.cloud import storage
import os
from sat2plan.scripts.params import BUCKET_NAME
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import shutil


def is_valid_map_image(img_path, min_std_threshold=0.015):
    try:
        image = Image.open(img_path)
        w = image.width
        # Extraire la partie plan (droite)
        map_img = image.crop((w//2, 0, w, image.height))
        # Convertir en array numpy et calculer l'écart-type
        map_array = np.array(map_img)
        # Calculer l'écart-type pour chaque canal et prendre la moyenne
        std_dev = np.mean([np.std(map_array[:,:,i]) for i in range(3)])
        # Normaliser par rapport à la plage de valeurs possibles (255 pour les images 8-bit)
        normalized_std = std_dev / 255.0
        is_valid = normalized_std > min_std_threshold
        if not is_valid:
            # Créer le répertoire removed s'il n'existe pas
            removed_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'removed')
            os.makedirs(removed_dir, exist_ok=True)
            # Déplacer l'image dans le répertoire removed
            new_path = os.path.join(removed_dir, os.path.basename(img_path))
            shutil.move(img_path, new_path)
            print(f"Moving {img_path} to removed: insufficient urban features (std_dev: {normalized_std:.3f})")
        return is_valid
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        if os.path.exists(img_path):
            # En cas d'erreur, déplacer aussi dans removed
            removed_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'removed')
            os.makedirs(removed_dir, exist_ok=True)
            new_path = os.path.join(removed_dir, os.path.basename(img_path))
            shutil.move(img_path, new_path)
        return False


def download_file(blob, file_path):
    if not os.path.exists(file_path):
        blob.download_to_filename(file_path)
        if is_valid_map_image(file_path):
            print(f"Downloaded and validated {file_path} from bucket {BUCKET_NAME}")
        else:
            print(f"Downloaded but invalid {file_path} from bucket {BUCKET_NAME}")


def download_bucket_folder(folder_name, val_size=0):
    destination_folder = './data/'
    if val_size == 0:
        os.makedirs(destination_folder + folder_name, exist_ok=True)
    else:
        os.makedirs(destination_folder + '/split/train/' +
                    folder_name, exist_ok=True)
        os.makedirs(destination_folder + '/split/val/' +
                    folder_name, exist_ok=True)
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=folder_name)

    with ThreadPoolExecutor(max_workers=32) as executor:
        for idx, blob in enumerate(blobs):
            if val_size == 0:
                file_path = os.path.join(destination_folder, blob.name)
            else:
                if idx % 10 > 10*val_size:
                    file_path = os.path.join(
                        destination_folder, 'split/train', blob.name)
                else:
                    file_path = os.path.join(
                        destination_folder, 'split/val', blob.name)
            executor.submit(download_file, blob, file_path)
