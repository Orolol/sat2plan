import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sat2plan.logic.preproc.data import download_bucket_folder
from sat2plan.logic.models.unet.model_training import train_model


# @mlflow_run
def train_unet():

    data_bucket = 'data-10k'

    print(Fore.YELLOW + "Training unet" + Style.RESET_ALL)
    download_bucket_folder(data_bucket, val_size=0.1)

    print("Running unet training")
    train_model(data_bucket=data_bucket)
