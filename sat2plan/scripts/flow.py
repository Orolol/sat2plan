import glob
import os
import time
import pickle
import torch
import shutil
import json
from datetime import datetime

from typing import Dict

from colorama import Fore, Style
from google.cloud import storage

from sat2plan.scripts.params import MODEL_TARGET, BUCKET_NAME, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MLFLOW_MODEL_NAME
from sat2plan.logic.configuration.config import Global_Configuration

import mlflow
from mlflow.tracking import MlflowClient

LOCAL_REGISTRY_PATH = os.getcwd() + "/checkpoints"


def check_disk_space(path, required_space_gb=5):
    """Vérifie s'il y a assez d'espace disque"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free // (2**30)
        return free_gb >= required_space_gb
    except Exception as e:
        print(f"Erreur lors de la vérification de l'espace disque: {e}")
        return False


def save_results(params: dict, metrics: dict) -> None:
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(
        LOCAL_REGISTRY_PATH, "params"), exist_ok=True)
    if params is not None:
        params_path = os.path.join(
            LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
    os.makedirs(os.path.join(
        LOCAL_REGISTRY_PATH, "metrics"), exist_ok=True)
    if metrics is not None:
        metrics_path = os.path.join(
            LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(models: Dict[str, torch.nn.Module] = None, optimizers: Dict[str, torch.optim.Optimizer] = None, suffix='') -> None:
    """Sauvegarde les modèles et optimizers de façon sécurisée"""
    save_dir = "save/checkpoints"
    temp_dir = "save/checkpoints/temp"
    
    try:
        # Vérifier l'espace disque
        if not check_disk_space(save_dir):
            print("Attention: Espace disque insuffisant pour la sauvegarde")
            return False

        # Créer les répertoires nécessaires
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Sauvegarder d'abord dans un fichier temporaire
        temp_path = os.path.join(temp_dir, f"model{suffix}_temp.pt")
        final_path = os.path.join(save_dir, f"model{suffix}.pt")

        # Préparer le dictionnaire de sauvegarde
        save_dict = {
            'gen_state_dict': models['gen'].module.state_dict() if hasattr(models['gen'], 'module') 
                            else models['gen'].state_dict(),
            'disc_state_dict': models['disc'].module.state_dict() if hasattr(models['disc'], 'module')
                             else models['disc'].state_dict(),
            'gen_opt_optimizer_state_dict': optimizers['gen_opt'].state_dict(),
            'gen_disc_optimizer_state_dict': optimizers['gen_disc'].state_dict(),
            'timestamp': datetime.now().isoformat()
        }

        # Sauvegarder dans le fichier temporaire
        torch.save(save_dict, temp_path)

        # Si la sauvegarde temporaire a réussi, déplacer vers l'emplacement final
        if os.path.exists(temp_path):
            shutil.move(temp_path, final_path)
            print(f"Modèle sauvegardé avec succès: {final_path}")
            return True
        else:
            print("Erreur: Le fichier temporaire n'a pas été créé")
            return False

    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle: {e}")
        # Nettoyer les fichiers temporaires en cas d'erreur
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def load_pred_model() -> torch.nn.Module:
    local_model_directory = os.path.join("pred_models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    print(f"Most recent model path: {most_recent_model_path_on_disk}")
    model = torch.load(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return model['gen_state_dict']


def load_model(stage="Production") -> torch.nn.Module:
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        print(f"Most recent model path: {most_recent_model_path_on_disk}")
        epoch = most_recent_model_path_on_disk.split("/")[-1].split("-")[-2]
        model = torch.load(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return (model, epoch)

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(
                LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            model = torch.load(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE +
              f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(
                name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(
                f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

            return None

        model = mlflow.pytorch.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None


def mlflow_run(func):
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.pytorch.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=Global_Configuration().device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
