import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sat2plan.logic.preproc.data import download_bucket_folder
from sat2plan.logic.preproc.sauvegarde_params import export_params_txt
from sat2plan.logic.models.unet.unet import Unet
from sat2plan.logic.models.dcgan import dcgan


# @mlflow_run
def train_unet():

    data_bucket = 'data-1k'
    # export_params_txt()

    print(Fore.YELLOW + "Training unet" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.1)

    print("Running unet training")
    # train_model(data_bucket=data_bucket)
    unet = Unet(data_bucket=data_bucket)
    unet.train()

# @mlflow_run
def train_dcgan():

    data_bucket = 'data-10k'
    # export_params_txt()

    print(Fore.YELLOW + "Training dcgan" + Style.RESET_ALL)
    download_bucket_folder(data_bucket)

    print("Running dcgan training")
    # train_model(data_bucket=data_bucket)
    dcgan.run_dcgan()


def pred():
    # TODO
    pass


def preprocess():
    # TODO
    pass


def download():
    # TODO
    pass


def evaluate():
    # TODO
    pass
