from google.cloud import storage
import os
from sat2plan.scripts.params import BUCKET_NAME


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
        if not os.path.exists(file_path):
            blob.download_to_filename(file_path)
            print(f"Downloaded {file_path} from bucket {BUCKET_NAME}")
