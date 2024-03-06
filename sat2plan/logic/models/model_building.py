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
    def __init__(self, img_size=256, kernel_size=3, stride=1, padding=1, channels=3):
        self.img_size = img_size
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding), nn.LeakyReLU(
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

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


#Générateur
class Generator(nn.Module):
    def __init__(self, img_size=256, latent_dim=100, kernel_size=3, stride=1, padding=1, channels=3):
        super(Generator, self).__init__()

        # self.init_size = img_size // 4
        self.init_size = img_size

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(channels, 128, kernel_size, stride, padding),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size, stride, padding),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size, stride, padding),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size, stride, padding),
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
