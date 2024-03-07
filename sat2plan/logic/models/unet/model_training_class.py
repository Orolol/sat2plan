import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sat2plan.logic.models.unet.global_config import Global_Configuration
from sat2plan.logic.models.unet.model_config import Model_Configuration

from sat2plan.logic.models.unet.model_building import Generator, Discriminator

from sat2plan.logic.models.unet.dataset import Satellite2Map_Data

from sat2plan.scripts.flow import save_results, save_model


class Model_Training():

    def __init__(self, data_bucket='data-1k'):
        # Import des paramètres globaux
        G_CFG = Global_Configuration()

        self.n_cpu = G_CFG.n_cpu

        self.device = G_CFG.device
        self.train_dir = f"{G_CFG.train_dir}/{data_bucket}"
        self.val_dir = f"{G_CFG.val_dir}/{data_bucket}"

        self.image_size = G_CFG.image_size

        self.batch_size = G_CFG.batch_size
        self.n_epochs = G_CFG.n_epochs
        self.sample_interval = G_CFG.sample_interval

        self.num_workers = G_CFG.num_workers
        self.l1_lambda = G_CFG.l1_lambda
        self.lambda_gp = G_CFG.lambda_gp

        self.load_model = G_CFG.load_model
        self.save_model_bool = G_CFG.save_model

        self.checkpoint_disc = G_CFG.checkpoint_disc
        self.checkpoint_gen = G_CFG.checkpoint_gen

        # Import des hyperparamètres du modèle
        M_CFG = Model_Configuration()

        self.learning_rate = M_CFG.learning_rate
        self.beta1 = M_CFG.beta1
        self.beta2 = M_CFG.beta2

        # Création des models, optimizers, losses
        self.create_models()

        # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
        torch.backends.cudnn.benchmark = True

        # Loading Data
        self.dataloading()

    # Create models, optimizers ans losses

    def create_models(self):
        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            print("Cuda is available")
            self.netD = Discriminator(in_channels=3).cuda()
            self.netG = Generator(in_channels=3).cuda()
        else:
            print("Cuda is not available")
            self.netD = Discriminator(in_channels=3)
            self.netG = Generator(in_channels=3)

        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

        self.BCE_Loss = nn.BCEWithLogitsLoss()
        self.L1_Loss = nn.L1Loss()
        self.Gen_loss = []
        self.Dis_loss = []
        return

    # Load datasets from train/val directories
    def dataloading(self):

        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        self.train_dataset = Satellite2Map_Data(root=self.train_dir)
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                   shuffle=True, pin_memory=True, num_workers=self.num_workers)
        print("Train Data Loaded")

        self.val_dataset = Satellite2Map_Data(root=self.val_dir)
        self.val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                 shuffle=True, pin_memory=True, num_workers=self.num_workers)
        print("Validation Data Loaded")

        return

    # Train & save models
    def train(self):
        for epoch in range(self.n_epochs):
            for idx, (x, y) in enumerate(self.train_dl):

                if cuda:
                    x = x .cuda()
                    y = y.cuda()

                ############## Train Discriminator ##############

                # Measure discriminator's ability to classify real from generated samples
                y_fake = self.netG(x)
                D_real = self.netD(x, y)
                D_real_loss = self.BCE_Loss(D_real, torch.ones_like(D_real))
                D_fake = self.netD(x, y_fake.detach())
                D_fake_loss = self.BCE_Loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss)/2

                # Backward and optimize
                self.netD.zero_grad()
                self.Dis_loss.append(D_loss.item())
                D_loss.backward()
                self.OptimizerD.step()

                ############## Train Generator ##############

                # Loss measures generator's ability to fool the discriminator
                D_fake = self.netD(x, y_fake)
                G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                G_loss = G_fake_loss + L1
                self.Gen_loss.append(G_loss.item())

                # Backward and optimize
                self.OptimizerG.zero_grad()
                G_loss.backward()
                self.OptimizerG.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, self.n_epochs, idx+1, len(self.train_dl), D_loss.item(), G_loss.item())
                )

                batches_done = epoch * len(self.train_dl) + idx
                if batches_done % self.sample_interval == 0:
                    concatenated_images = torch.cat(
                        (x[:-1], y_fake[:-1], y[:-1]), dim=2)

                    save_image(concatenated_images, "images/%d.png" %
                               batches_done, nrow=3, normalize=True)

            if self.save_model_bool and (epoch+1) % 5 == 0:
                save_model(self.netG)
                save_model(self.netD)
                save_results(params=self.M_CFG, metrics=dict(
                    Gen_loss=self.Gen_loss, Dis_loss=self.Dis_loss))

        save_results(params=self.M_CFG, metrics=dict(
            Gen_loss=self.Gen_loss, Dis_loss=self.Dis_loss))

        return
