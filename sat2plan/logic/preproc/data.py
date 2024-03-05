from google.cloud import storage
import os
from sat2plan.scripts.params import BUCKET_NAME


def download_bucket_folder(folder_name):
    destination_folder = './sat2plan/data/'
    os.makedirs(destination_folder + folder_name, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=folder_name)

    for blob in blobs:
        file_path = os.path.join(destination_folder, blob.name)
        if not os.path.exists(file_path):
            blob.download_to_filename(file_path)
            print(f"Downloaded {file_path} from bucket {BUCKET_NAME}")
