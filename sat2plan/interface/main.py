import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from matplotlib import pyplot as plt
from PIL import Image

from sat2plan.logic.preproc.data import download_bucket_folder
from sat2plan.logic.preproc.sauvegarde_params import export_params_txt
from sat2plan.logic.models.ucvgan.ucvgan import UCVGan, Generator
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


def pred() -> np.ndarray:
    pred_model = None
    if pred_model is None:
        pred_model = load_pred_model()

    image_satellite = np.asarray(Image.open(
        "data/adresse/adresse_satellite.jpg").convert('RGB'))

    # convert image to tensor
    image_satellite = image_satellite / 255
    image_satellite = np.transpose(image_satellite, (2, 0, 1))
    image_satellite = np.expand_dims(image_satellite, axis=0)
    image_satellite = torch.tensor(image_satellite, dtype=torch.float32)

    netG = Generator(in_channels=3)

    netG.load_state_dict(pred_model)
    y_pred = netG(image_satellite)
    save_image(
        y_pred, f"data/adresse/adresse_generee.png", normalize=True)
    return y_pred


def evaluate():
    # TODO
    pass
