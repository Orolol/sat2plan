import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sat2plan.logic.preproc.data import download_bucket_folder
from sat2plan.logic.preproc.sauvegarde_params import export_params_txt
from sat2plan.logic.models.unet.unet import Unet
<<<<<<< HEAD
from sat2plan.logic.models.dcgan.dcgan import Dcgan
=======
from sat2plan.logic.models.samgan.samgan import SAMGAN
>>>>>>> 605a57e1026dadba72c2909bea7ed61a5f8db64f


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
def train_dcgan():

    data_bucket = 'data-10k'
    # export_params_txt()

    print(Fore.YELLOW + "Training dcgan" + Style.RESET_ALL)
    download_bucket_folder(data_bucket)

    print("Running dcgan training")
    # train_model(data_bucket=data_bucket)
    dcgan = Dcgan(data_bucket=data_bucket)
    dcgan.train()


def train_sam_gan():

    data_bucket = 'data-10k'
    # export_params_txt()

    print(Fore.YELLOW + "Training sam_gan" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.2)

    print("Running sam_gan training")
    # train_model(data_bucket=data_bucket)
    sam_gan = SAMGAN(data_bucket=data_bucket)
    sam_gan.train()


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
