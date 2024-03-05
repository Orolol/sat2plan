import os


DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
