import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from sat2plan.logic.models.unet.model_building import Generator, Discriminator
from sat2plan.logic.models.unet.model_config import Model_Configuration
from sat2plan.scripts.flow import save_results, save_model, mlflow_run

from sat2plan.logic.models.unet.dataset import Satellite2Map_Data


# @mlflow_run
def train_model(data_bucket='data-1k'):
    # Import des hyperparamètres
    CFG = Model_Configuration()

    device = CFG.device
    train_dir = f"{CFG.train_dir}/{data_bucket}"
    val_dir = f"{CFG.val_dir}/{data_bucket}"

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
    if cuda:
        print("Cuda is available")
        netD = Discriminator(in_channels=3).cuda()
        netG = Generator(in_channels=3).cuda()
    else:
        print("Cuda is not available")
        netD = Discriminator(in_channels=3)
        netG = Generator(in_channels=3)
    OptimizerD = torch.optim.Adam(
        netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
    OptimizerG = torch.optim.Adam(
        netG.parameters(), lr=learning_rate, betas=(beta1, beta2))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    torch.backends.cudnn.benchmark = True
    Gen_loss = []
    Dis_loss = []

    os.makedirs("images", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    train_dataset = Satellite2Map_Data(root=train_dir)
    train_dl = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, pin_memory=True, num_workers=num_workers)
    print("Train Data Loaded")

    val_dataset = Satellite2Map_Data(root=val_dir)
    val_dl = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)

    for epoch in range(n_epochs):
        for idx, (x, y) in enumerate(train_dl):
            if cuda:
                x = x .cuda()
                y = y.cuda()

            ############## Train Discriminator ##############

            # Measure discriminator's ability to classify real from generated samples
            y_fake = netG(x)
            D_real = netD(x, y)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = netD(x, y_fake.detach())
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2

            # Backward and optimize
            netD.zero_grad()
            Dis_loss.append(D_loss.item())
            D_loss.backward()
            OptimizerD.step()

            ############## Train Generator ##############

            # Loss measures generator's ability to fool the discriminator
            D_fake = netD(x, y_fake)
            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L1 = L1_Loss(y_fake, y) * l1_lambda
            G_loss = G_fake_loss + L1
            Gen_loss.append(G_loss.item())

            # Backward and optimize
            OptimizerG.zero_grad()
            G_loss.backward()
            OptimizerG.step()

            batches_done = epoch * len(train_dl) + idx
            if batches_done % sample_interval == 0:
                concatenated_images = torch.cat(
                    (x[:-1], y_fake[:-1], y[:-1]), dim=2)

                save_image(concatenated_images, "images/%d.png" %
                           batches_done, nrow=3, normalize=True)
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, n_epochs, D_loss.item(), G_loss.item())
        )
        if save_model_bool and (epoch+1) % 5 == 0:
            save_model({"gen": netG, "disc": netD}, {
                       "gen_opt": OptimizerG, "gen_disc": OptimizerD}, suffix=f"-{epoch}-G")
            save_results(params=CFG, metrics=dict(
                Gen_loss=Gen_loss, Dis_loss=Dis_loss))
    save_results(params=CFG, metrics=dict(
        Gen_loss=Gen_loss, Dis_loss=Dis_loss))
