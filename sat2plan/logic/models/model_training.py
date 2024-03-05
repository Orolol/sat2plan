from model_building import Generator, Discriminator, weights_init_normal
from model_config import Configuration

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch

import os

#Import des hyperparamètres
HPT = Configuration()

n_epochs = HPT.n_epochs
batch_size = HPT.batch_size
lr = HPT.lr
b1 = HPT.b1
b2 = HPT.b2
n_cpu = HPT.n_cpu
latent_dim = HPT.latent_dim
img_size = HPT.img_size
n_blocks = HPT.n_blocks
stride = HPT.stride
padding = HPT.padding
kernel_size = HPT.kernel_size
channels = HPT.channels
sample_interval = HPT.sample_interval
from_scratch = True
cuda = True if torch.cuda.is_available() else False


os.makedirs("images", exist_ok=True)


#Loss
adversarial_loss = nn.BCELoss()


#Initialisation du discriminateur et du générateur
discriminator = Discriminator(img_size, n_blocks, stride, padding, kernel_size, channels)
generator = Generator(img_size, latent_dim, n_blocks, stride, padding, kernel_size, channels)

if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

#Initialisation des poids
discriminator.apply(weights_init_normal)
generator.apply(weights_init_normal)


# Configuration data loader
dataloader = DataLoader(
    datasets.ImageFolder("sat2plan/data", transform=transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])),
    batch_size=batch_size,
    shuffle=True
    )


# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        sat = F.interpolate(imgs[:, :, :, :512],
                            size=(img_size, img_size))
        plan = F.interpolate(imgs[:, :, :, 512:],
                                size=(img_size, img_size))


        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(
            0.0), requires_grad=False)

        # Configure input
        real_imgs = plan

        # ---------------------
        #  Train discriminateur
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(
            discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # Sauvegarder l'image
        # save_image(concatenated_images,
        #            f'gen_images/concatenated_image{epoch}-{i} .jpg')

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            concatenated_images = torch.cat(
                (gen_imgs[:-5], sat[:-5], real_imgs[:-5]), dim=2)

            save_image(concatenated_images, "images/%d.png" %
                        batches_done, nrow=5, normalize=True)

        # -----------------
        #  Train générateur
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(sat)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()


    torch.save(generator.state_dict(),
                f'models_checkpoint/generator.pth')
    torch.save(discriminator.state_dict(),
                f'models_checkpoint/discriminator.pth')
