# Script to init Discriminator & Generator

# Torch Import
import torch
import torch.nn as nn

####
# Creating our classes
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input Shape : N x channels_img, 32 x 32
            nn.Conv2d(
                in_channels= channels_img,
                out_channels= features_d,
                kernel_size= 4,
                stride= 2,
                padding= 1
            ),

            nn.LeakyReLU(.2 ),
            self.__block(in_channels= features_d, out_channels= features_d * 2 , kernel_size= 4, stride= 2, padding= 1), # 32 x 32
            self.__block(in_channels= features_d * 2, out_channels= features_d * 4 , kernel_size= 4, stride= 2, padding= 1), # 16 x 16
            self.__block(in_channels= features_d * 4, out_channels= features_d * 8 , kernel_size= 4, stride= 2, padding= 1), # 8 x 8
            nn.Conv2d(in_channels= features_d * 8, out_channels= 1, kernel_size= 4, stride= 2, padding= 0), # 1 x 1
            nn.Sigmoid()
        )

    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels= in_channels,
                out_channels= out_channels,
                kernel_size= kernel_size,
                stride= stride,
                padding= padding,
                bias= False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

    def forward(self, x):
        return self.disc(x)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input : N x z_dim x 1 x 1
            self.__block(in_channels= z_dim, out_channels= features_g * 16, kernel_size= 4, stride= 1, padding= 0),  # N x f_g * 16 x 4 x 4
            self.__block(in_channels= features_g * 16, out_channels= features_g * 8, kernel_size= 4, stride= 2, padding= 1), # 8 x 8
            self.__block(in_channels= features_g * 8, out_channels= features_g * 4, kernel_size= 4, stride= 2, padding= 1), # 16 x 16
            self.__block(in_channels= features_g * 4, out_channels= features_g * 2, kernel_size= 4, stride= 2, padding= 1), # 32 x 32
            nn.ConvTranspose2d(
                in_channels= features_g * 2,
                out_channels= channels_img,
                kernel_size= 4,
                stride= 2,
                padding= 1
            ),
            nn.Tanh() # [-1, 1]
        )

    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels= in_channels,
                out_channels= out_channels,
                kernel_size= kernel_size,
                stride= stride,
                padding= padding,
                bias= False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

# Function to init our weights depending the layer
def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, .0, .02)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, .0, .02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, .0, .02)

# Just testing the good shape output froom our classes
def test_gen_disc_shape():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channels, H, W))
    discriminator = Discriminator(in_channels, 8)
    initialize_weights(discriminator)

    print("Testing Discrimitnator ... ")
    assert discriminator(x).shape == (N, 1, 1, 1)

    z = torch.randn((N, z_dim, 1, 1))
    generator = Generator(z_dim, in_channels, 8)
    initialize_weights(generator)

    print("Testing Generator ... ")
    assert generator(z).shape == (N, in_channels, H, W)

    print("Sucess ✅")

if __name__ == '__main__':
    test_gen_disc_shape()
