import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sat2plan.logic.models.basegan.dcgan import run_dcgan
from sat2plan.logic.preproc.data import download_bucket_folder


# @mlflow_run
def train():
    print(Fore.YELLOW + "Training DCGAN" + Style.RESET_ALL)
    print("Downloading data from GCP bucket")
    download_bucket_folder('data-1k')
    print("Running DCGAN")
    run_dcgan()


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
