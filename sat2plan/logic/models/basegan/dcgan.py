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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=6,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10,
                    help="interval between image sampling")
opt = parser.parse_args()
parser.add_argument("--from_scratch", type=bool, default=False,
                    help="use model from scratch or not")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.init_size = opt.img_size // 4
        self.init_size = opt.img_size

        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(opt.channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # print("Z", z.shape)
        # out = self.l1(z)
        # out = z.view(z.shape[0], 128, self.init_size, self.init_size)
        # print("OUT", z.shape)
        img = self.conv_blocks(z)
        # print("IMG", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
if not opt.from_scratch:

    generator_files = glob.glob('models_checkpoint/generator_*.pth')
    discriminator_files = glob.glob('models_checkpoint/discriminator_*.pth')

    generator_files.sort(key=os.path.getmtime, reverse=True)
    discriminator_files.sort(key=os.path.getmtime, reverse=True)

    if generator_files:
        print(generator_files[0])
        generator.load_state_dict(torch.load(generator_files[0]))
    if discriminator_files:
        discriminator.load_state_dict(torch.load(discriminator_files[0]))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder("../../../../data/data-1k", transform=transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])),
    batch_size=opt.batch_size,
    shuffle=True,

)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------


for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        sat = F.interpolate(imgs[:, :, :, :512],
                            size=(opt.img_size, opt.img_size))
        plan = F.interpolate(imgs[:, :, :, 512:],
                             size=(opt.img_size, opt.img_size))

        # plt.imshow(plan[0].permute(1, 2, 0))
        # plt.show()
        # plt.imshow(sat[0].permute(1, 2, 0))
        # plt.show()

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(
            0.0), requires_grad=False)

        # Configure input
        real_imgs = plan

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(sat)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # Sauvegarder l'image
        # save_image(concatenated_images,
        #            f'gen_images/concatenated_image{epoch}-{i} .jpg')

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            concatenated_images = torch.cat(
                (gen_imgs[:-5], sat[:-5], real_imgs[:-5]), dim=2)

            save_image(concatenated_images, "images/%d.png" %
                       batches_done, nrow=5, normalize=True)
    torch.save(generator.state_dict(),
               f'models_checkpoint/generator.pth')
    torch.save(discriminator.state_dict(),
               f'models_checkpoint/discriminator.pth')
