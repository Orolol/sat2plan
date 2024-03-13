import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from matplotlib import pyplot as plt
from PIL import Image

from sat2plan.logic.preproc.data import download_bucket_folder
from sat2plan.logic.preproc.sauvegarde_params import export_params_txt
from sat2plan.logic.models.ucvgan.ucvgan import UCVGan
from sat2plan.logic.models.unet.unet import Unet
from sat2plan.logic.models.samgan.samgan import SAMGAN
from sat2plan.scripts.flow import load_pred_model

pred_model = None

# @mlflow_run


def train_unet():

    data_bucket = 'data-10k'
    # export_params_txt()

    print(Fore.YELLOW + "Training unet" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.1)

    print("Running unet training")
    # train_model(data_bucket=data_bucket)
    unet = Unet(data_bucket=data_bucket)
    unet.train()

# @mlflow_run


def train_ucvgan():

    data_bucket = 'data-1k'
    # export_params_txt()

    print(Fore.YELLOW + "Training UCVGan" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.05)

    print("Running unet training")
    # train_model(data_bucket=data_bucket)
    ucvgan = UCVGan(data_bucket=data_bucket)
    ucvgan.train()


def train_sam_gan():

    data_bucket = 'data-10k'
    # export_params_txt()

    print(Fore.YELLOW + "Training sam_gan" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.2)

    print("Running sam_gan training")
    # train_model(data_bucket=data_bucket)
    sam_gan = SAMGAN(data_bucket=data_bucket)
    sam_gan.train()


def pred(image_satellite_path="images_pred/000004_San_Jose_37.23808_-121.91478.png") -> np.ndarray:
    pred_model = None
    if pred_model is None:
        pred_model = load_pred_model()

    image_satellite = np.asarray(Image.open(
        image_satellite_path).convert('RGB'))

    y_pred = pred_model(image_satellite)
    plt.imshow(y_pred)
    plt.show()
    return y_pred


def evaluate():
    # TODO
    pass
