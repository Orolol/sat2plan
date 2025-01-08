import os
import torch
import torch.multiprocessing as mp
from sat2plan.logic.models.ucvgan.ucvgan import UCVGan
from sat2plan.logic.models.samgan.samgan import SAMGAN
from sat2plan.logic.models.unet.unet import Unet
from sat2plan.logic.models.basegan.dcgan import run_dcgan
from sat2plan.logic.configuration.config import Global_Configuration
from sat2plan.logic.preproc.dataset import Satellite2Map_Data

def train_ucvgan():
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(UCVGan,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_samgan():
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(SAMGAN,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_unet():
    G_CFG = Global_Configuration()
    world_size = torch.cuda.device_count()
    print("Number of GPUs:", world_size)
    mp.spawn(Unet,
            args=(world_size,),
            nprocs=world_size,
            join=True)

def train_dcgan():
    run_dcgan()
