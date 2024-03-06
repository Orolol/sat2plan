import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from model_building import Generator, Discriminator, weights_init_normal
from model_config import Configuration
from sat2plan.scripts.flow import save_results, save_model, mlflow_run

from dataset import Satellite2Map_Data


@mlflow_run
def train_model():
    # Import des hyperparamètres
    CFG = Configuration()

    device = CFG.device
    train_dir = CFG.train_dir
    val_dir = CFG.val_dir

    learning_rate = CFG.learning_rate
    beta1 = CFG.beta1
    beta2 = CFG.beta2

    n_cpu = CFG.n_cpu

    batch_size = CFG.batch_size
    n_epochs = CFG.n_epochs
    sample_interval = CFG.sample_interval

    image_size = CFG.image_size

    num_workers = CFG.num_workers
    l1_lambda = CFG.l1_lambda
    lambda_gp = CFG.lambda_gp

    load_model = CFG.load_model
    save_model_bool = CFG.save_model

    checkpoint_disc = CFG.checkpoint_disc
    checkpoint_gen = CFG.checkpoint_gen

    cuda = True if torch.cuda.is_available() else False

    netD = Discriminator(in_channels=3).cuda()
    netG = Generator(in_channels=3).cuda()
    OptimizerD = torch.optim.Adam(
        netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
    OptimizerG = torch.optim.Adam(
        netG.parameters(), lr=learning_rate, betas=(beta1, beta2))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.backends.cudnn.benchmark = True
    Gen_loss = []
    Dis_loss = []

    os.makedirs("images", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    train_dataset = Satellite2Map_Data(root=config.TRAIN_DIR)
    train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # G_Scaler = torch.cuda.amp.GradScaler()
    # D_Scaler = torch.cuda.amp.GradScaler()
    val_dataset = Satellite2Map_Data(root=config.VAL_DIR)
    val_dl = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    for epoch in range(n_epochs):
        for idx, (x, y) in enumerate(train_dl):
            print(f"Batch : {idx+1}/{len(train_dl)}")

            ############## Train Discriminator ##############
            # with torch.cuda.amp.autocast():
            y_fake = netG(x)
            D_real = netD(x, y)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = netD(x, y_fake.detach())
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2
            netD.zero_grad()
            Dis_loss.append(D_loss.item())
            D_loss.backward()
            # D_Scaler.scale(D_loss).backward()
            OptimizerD.step()
            # D_Scaler.step(OptimizerD)
            # D_Scaler.update()

            ############## Train Generator ##############
            # with torch.cuda.amp.autocast():

            D_fake = netD(x, y_fake)
            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L1 = L1_Loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            OptimizerG.zero_grad()
            Gen_loss.append(G_loss.item())
            G_loss.backward()
            # G_Scaler.scale(G_loss).backward()
            # G_Scaler.step(OptimizerG)
            OptimizerG.step()
            # G_Scaler.update()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, n_epochs, i+1, len(dataloader), D_loss.item(), G_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                concatenated_images = torch.cat(
                    (sat[:-1], y_fake[:-1], real_imgs[:-1]), dim=2)

                save_image(concatenated_images, "images/%d.png" %
                           batches_done, nrow=3, normalize=True)

        if save_model_bool and (epoch+1) % 5 == 0:
            save_model(netG, netD, OptimizerG, OptimizerD, epoch, CFG)

    save_results(params=CFG, metrics=dict(
        Gen_loss=Gen_loss, Dis_loss=Dis_loss))
