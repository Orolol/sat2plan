import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import math
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from sat2plan.logic.models.basegan.dataset import Satellite2Map_Data


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size=512, latent_dim=100, channels=3):
        super(Generator, self).__init__()

        # self.init_size = img_size // 4
        self.init_size = img_size

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(latent_dim, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # print("Z", z.shape)
        # out = self.l1(z)
        # out = z.view(z.shape[0], 128, self.init_size, self.init_size)
        #print("OUT", z.shape)
        img = self.conv_blocks(z)
        #print("IMG", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size=512, channels=3):
        self.img_size = img_size
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, y):
        img = torch.cat([img, y], dim=1)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def run_dcgan():

    os.makedirs("images", exist_ok=True)
    os.makedirs("models_checkpoint", exist_ok=True)

    n_epochs = 600
    batch_size = 1
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 6
    latent_dim = 100
    img_size = 512
    channels = 3
    sample_interval = 10
    from_scratch = True

    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(img_size=img_size)
    discriminator = Discriminator(img_size=img_size)
    if not from_scratch:

        generator_files = glob.glob('models_checkpoint/generator_*.pth')
        discriminator_files = glob.glob(
            'models_checkpoint/discriminator_*.pth')

        generator_files.sort(key=os.path.getmtime, reverse=True)
        discriminator_files.sort(key=os.path.getmtime, reverse=True)

        if generator_files:
            print(generator_files[0])
            generator.load_state_dict(torch.load(generator_files[0]))
        if discriminator_files:
            discriminator.load_state_dict(torch.load(discriminator_files[0]))
    print("Check CUDA")
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    print("Check CUDA :", cuda)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    os.makedirs("data", exist_ok=True)

    train_dir = "./data/split/train/data-10k"

    train_dataset = Satellite2Map_Data(root=train_dir)
    train_dl = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True, num_workers=2)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    netG = Generator().cuda().apply(weights_init_normal)
    netD = Discriminator().cuda().apply(weights_init_normal)
    BCE_Loss = nn.BCEWithLogitsLoss().cuda()
    OptimizerD = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(b1, b2))
    OptimizerG = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(b1, b2))
    Dis_loss = []
    Gen_loss = []

    # ----------
    #  Training
    # ----------
    print('Start training')
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(train_dl):

            if cuda:
                x = x.cuda()
                y = y.cuda()

            sat = x
            plan = y

            # plt.imshow(plan[0].permute(1, 2, 0))
            # plt.show()
            # plt.imshow(sat[0].permute(1, 2, 0))
            # plt.show()

            # Adversarial ground truths
            '''
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                0.0), requires_grad=False)
            '''
            # Configure input
            real_imgs = plan

            # -----------------
            #  Train Generator
            # -----------------

            '''optimizer_G.zero_grad()
            # Generate a batch of images
            try:
                gen_imgs = generator(sat)
            except Exception as e:
                print("Error", e)

            # Loss measures generator's ability to fool the discriminator
            D_fake = discriminator(gen_imgs).detach()
            g_loss = Variable(adversarial_loss(D_fake, torch.ones_like(D_fake)), requires_grad=True)
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            D_real = discriminator(real_imgs).detach()
            real_loss = adversarial_loss(D_real, torch.ones_like(D_real))
            fake_loss = adversarial_loss(D_fake, torch.zeros_like(D_fake))
            d_loss = Variable((real_loss + fake_loss) / 2, requires_grad=True)
            d_loss.backward()
            optimizer_D.step()'''

            ############## Train Discriminator ##############

            # Measure discriminator's ability to classify real from generated samples
            y_fake = netG(sat)
            D_real = netD(sat, real_imgs)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = netD(sat, y_fake.detach())
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2

            # Backward and optimize
            netD.zero_grad()
            Dis_loss.append(D_loss.item())
            D_loss.backward()
            OptimizerD.step()

            ############## Train Generator ##############

            # Loss measures generator's ability to fool the discriminator
            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            Gen_loss.append(G_fake_loss.item())

            # Backward and optimize
            OptimizerG.zero_grad()
            G_fake_loss.backward()
            OptimizerG.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, n_epochs, i+1, len(train_dl), D_loss.item(), G_fake_loss.item())
            )

            # Sauvegarder l'image
            # save_image(concatenated_images,
            #            f'gen_images/concatenated_image{epoch}-{i} .jpg')

            batches_done = epoch * len(train_dl) + i
            if batches_done % sample_interval == 0:
                concatenated_images = torch.cat(
                    (y_fake[:], sat[:], real_imgs[:]), dim=2)

                save_image(concatenated_images, "images/%d.png" %
                           batches_done, nrow=5, normalize=True)
        torch.save(generator.state_dict(),
                   f'models_checkpoint/generator.pth')
        torch.save(discriminator.state_dict(),
                   f'models_checkpoint/discriminator.pth')
