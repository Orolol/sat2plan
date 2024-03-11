import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchsummary import summary

from sat2plan.logic.models.ucvgan.global_config import Global_Configuration
from sat2plan.logic.models.ucvgan.model_config import Model_Configuration

from sat2plan.logic.models.ucvgan.model_building import Generator, Discriminator

from sat2plan.logic.models.ucvgan.dataset import Satellite2Map_Data

from sat2plan.scripts.flow import save_results, save_model, load_model

from sat2plan.logic.preproc.sauvegarde_params import ouverture_fichier_json, export_loss
# Modèle Unet


class UCVGan():
    def __init__(self, data_bucket='data-1k'):

        # Import des paramètres globaux
        self.G_CFG = Global_Configuration()

        self.n_cpu = self.G_CFG.n_cpu

        self.device = self.G_CFG.device
        self.train_dir = f"{self.G_CFG.train_dir}/{data_bucket}"
        self.val_dir = f"{self.G_CFG.val_dir}/{data_bucket}"

        self.image_size = self.G_CFG.image_size

        self.batch_size = self.G_CFG.batch_size
        self.n_epochs = self.G_CFG.n_epochs
        self.sample_interval = self.G_CFG.sample_interval

        self.num_workers = self.G_CFG.num_workers
        self.l1_lambda = self.G_CFG.l1_lambda
        self.lambda_gp = self.G_CFG.lambda_gp

        self.load_model = self.G_CFG.load_model
        self.save_model_bool = self.G_CFG.save_model

        self.checkpoint_disc = self.G_CFG.checkpoint_disc
        self.checkpoint_gen = self.G_CFG.checkpoint_gen

        # Import des hyperparamètres du modèle
        self.M_CFG = Model_Configuration()

        self.learning_rate = self.M_CFG.learning_rate
        self.beta1 = self.M_CFG.beta1
        self.beta2 = self.M_CFG.beta2

        # Loading Data
        self.dataloading()

        # Création des models, optimizers, losses
        self.create_models()

        # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
        torch.backends.cudnn.benchmark = True

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

    # Create models, optimizers ans losses

    def create_models(self):

        self.netD = Discriminator(in_channels=3)
        self.netG = Generator(in_channels=3)

        if self.load_model:
            model_and_optimizer = load_model()
            self.netG.load_state_dict(model_and_optimizer['gen_state_dict'])
            self.netD.load_state_dict(model_and_optimizer['disc_state_dict'])

        # Check Cuda
        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            print("Cuda is available")
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

        if self.load_model:
            model_and_optimizer = load_model()
            self.OptimizerG.load_state_dict(
                model_and_optimizer['gen_opt_optimizer_state_dict'])
            self.OptimizerD.load_state_dict(
                model_and_optimizer['gen_disc_optimizer_state_dict'])

        self.BCE_Loss = nn.BCEWithLogitsLoss()
        self.L1_Loss = nn.L1Loss()
        self.Gen_loss = []
        self.Dis_loss = []
        self.val_Dis_loss = []
        self.val_Gen_loss = []
        self.val_Gen_fake_loss = []
        self.val_Gen_L1_loss = []

        return

    # Train & save models
    def train(self):
        # Création du fichier params.json
        os.makedirs("save", exist_ok=True)
        params_json = open("params.json", mode="w", encoding='UTF-8')
        pytorch_total_params_G = sum(
            p.numel() for p in self.netG.parameters() if p.requires_grad)
        pytorch_total_params_D = sum(
            p.numel() for p in self.netD.parameters() if p.requires_grad)
        print("Total params in Generator :", pytorch_total_params_G)
        print("Total params in Discriminator :", pytorch_total_params_D)

        for epoch in range(self.n_epochs):
            for idx, (x, y, to_save) in enumerate(self.train_dl):

                if self.cuda:
                    x = x .cuda()
                    y = y.cuda()

                ############## Train Discriminator ##############

                # Measure discriminator's ability to classify real from generated samples
                y_fake = self.netG(x)
                if to_save:
                    save_image(y_fake, f"save/y_gen_{epoch}_{idx}.png")
                    save_image(x, f"save/input_{epoch}_{idx}.png")
                    save_image(y, f"save/label_{epoch}_{idx}.png")
                # print("NETD0", x.shape, y.shape)
                D_real = self.netD(x, y)
                D_real_loss = self.BCE_Loss(D_real, torch.ones_like(D_real))
                # print("NETD1", x.shape, y_fake.shape)
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
                # print("NETD", y_fake.shape, y.shape)
                D_fake = self.netD(x, y_fake)
                G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                G_loss = G_fake_loss + L1
                self.Gen_loss.append(G_loss.item())

                # Backward and optimize
                self.OptimizerG.zero_grad()
                G_loss.backward()
                self.OptimizerG.step()

                if idx % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch+1, self.n_epochs, idx+1, len(self.train_dl), D_loss.item(), G_loss.item())
                    )
                    concatenated_images = torch.cat(
                        (x[:], y_fake[:], y[:]), dim=2)

                    save_image(concatenated_images, "images/%d.png" %
                               str(epoch) + "-" + str(batches_done), nrow=3, normalize=True)

                # export_loss(params_json, epoch+1, idx+1, L1.item(), G_loss.item(), D_loss.item(), Global_Configuration())

                batches_done = epoch * len(self.train_dl) + idx

                if idx == 0:
                    concatenated_images = torch.cat(
                        (x[:], y_fake[:], y[:]), dim=2)

                    save_image(concatenated_images, "images/%d.png" %
                               epoch, nrow=3, normalize=True)

            if epoch != 0:
                print("-- Test de validation --")
                self.validation()
                print(f"Epoch : {epoch+1}/{self.n_epochs} :")
                print(
                    f"Validation Discriminator Loss : {self.val_Dis_loss[-1]}")
                print(
                    f"Validation Generator Loss : {self.val_Gen_loss[-1]} : {self.val_Gen_fake_loss[-1]} + {self.val_Gen_L1_loss[-1]}")
                print("------------------------")

            if self.save_model_bool:
                if epoch < 11 or (self.val_Gen_loss[-1] + self.val_Dis_loss[-1] < sum([x+y for x in self.val_Gen_loss[:-1] for y in self.val_Dis_loss[:-1]])/len(self.val_Gen_loss)):
                    save_model({"gen": self.netG, "disc": self.netD}, {
                        "gen_opt": self.OptimizerG, "gen_disc": self.OptimizerD}, suffix=f"-{epoch}-G")
                    save_results(params=self.M_CFG, metrics=dict(
                        Gen_loss=G_loss, Dis_loss=D_loss))

        save_model({"gen": self.netG, "disc": self.netD}, {
            "gen_opt": self.OptimizerG, "gen_disc": self.OptimizerD}, suffix=f"-{epoch}-G")
        save_results(params=self.M_CFG, metrics=dict(
            Gen_loss=G_loss, Dis_loss=D_loss))

        params_json.close()

        return

    # Test du modèle sur le set de validation
    def validation(self):

        sum_D_loss = 0
        sum_G_loss = 0
        sum_G_fake_loss = 0
        sum_G_L1_loss = 0

        for idx, (x, y) in enumerate(self.val_dl):
            ############## Discriminator ##############

            if self.cuda:
                x = x .cuda()
                y = y.cuda()

            # Measure discriminator's ability to classify real from generated samples
            y_fake = self.netG(x)
            D_real = self.netD(x, y)
            D_real_loss = self.BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = self.netD(x, y_fake.detach())
            D_fake_loss = self.BCE_Loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2

            sum_D_loss += D_loss.item()
            ############## Generator ##############

            # Loss measures generator's ability to fool the discriminator
            D_fake = self.netD(x, y_fake)
            G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
            G_L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
            G_loss = G_fake_loss + G_L1

            sum_G_loss += G_loss.item()
            sum_G_fake_loss += G_fake_loss.item()
            sum_G_L1_loss += G_L1.item()

            self.val_Dis_loss.append(sum_D_loss/len(self.val_dl))
            self.val_Gen_loss.append(sum_G_loss/len(self.val_dl))
            self.val_Gen_fake_loss.append(sum_G_fake_loss/len(self.val_dl))
            self.val_Gen_L1_loss.append(sum_G_L1_loss/len(self.val_dl))
