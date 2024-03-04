from google.cloud import storage
import os


def download_bucket_folder(bucket_name, folder_name):
    destination_folder = 'data/'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_name)

    for blob in blobs:
        file_path = os.path.join(destination_folder, blob.name)
        if not os.path.exists(file_path):
            blob.download_to_filename(file_path)
            print(f"Downloaded {file_path} from bucket {bucket_name}")
