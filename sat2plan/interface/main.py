import os
import torch
import torch.multiprocessing as mp
from sat2plan.logic.models.ucvgan.ucvgan import UCVGan
from sat2plan.logic.models.samgan.samgan import SAMGAN
from sat2plan.logic.models.unet.unet import Unet
from sat2plan.logic.models.basegan.dcgan import run_dcgan
from sat2plan.logic.configuration.config import Global_Configuration
from sat2plan.logic.preproc.dataset import Satellite2Map_Data
from sat2plan.logic.preproc.data import download_bucket_folder

def ensure_data_downloaded():
    G_CFG = Global_Configuration()
    if not os.path.exists(f"data/split/train/{G_CFG.data_bucket}") or \
       not os.path.exists(f"data/split/val/{G_CFG.data_bucket}"):
        print("Downloading dataset...")
        download_bucket_folder(G_CFG.data_bucket, val_size=0.2)
    else:
        print("Dataset already downloaded")

def train_ucvgan():
    ensure_data_downloaded()
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(UCVGan,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_samgan():
    ensure_data_downloaded()
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(SAMGAN,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_unet():
    ensure_data_downloaded()
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(Unet,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_dcgan():
    ensure_data_downloaded()
    run_dcgan()
