import torch.nn as nn
import torch

#Initialisation des poids
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#Discriminateur
class Discriminator(nn.Module):
    def __init__(self, img_size=256, n_blocks=4, stride=1, padding=1, kernel_size=3, channels=3):
        super(Discriminator, self).__init__()


        #Créé x blocks (Conv2d, BatchNorm, LeakyRelu) pour le discriminator
        def discriminator_block(block_num):
            block = []
            in_filters = channels
            out_filters = img_size // (2 ** (block_num - 1))

            for _ in range(block_num):
                block.append(nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding))
                block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.append(nn.LeakyReLU(0.2, inplace=True))
                in_filters = out_filters
                out_filters *= 2

            return block

        self.model = nn.Sequential(
            *discriminator_block(n_blocks),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear((img_size**2), 1),
            nn.Sigmoid()
            )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


#Générateur
class Generator(nn.Module):
    def __init__(self, img_size=256, latent_dim=100, n_blocks=4, stride=1, padding=1, kernel_size=3, channels=3):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, (img_size**2)))

        #Créé x blocks (ConvTranspose2d, BatchNorm2d, LeakyRelu) pour le generator
        def generator_block(block_num):
            block = []
            in_filters = channels
            out_filters = img_size

            for _ in range(block_num):
                block.append(nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding))
                block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.append(nn.LeakyReLU(0.2, inplace=True))
                in_filters = out_filters
                out_filters //= 2

            return block

        self.conv_blocks = nn.Sequential(
            *generator_block(n_blocks),
            nn.ConvTranspose2d(img_size, channels, kernel_size, stride, padding),
            nn.Tanh()
            )

    def forward(self, z):

        img = self.conv_blocks(z)

        return img
