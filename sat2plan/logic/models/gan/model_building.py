import torch
import torch.nn as nn
from sat2plan.logic.models.blocks.blocks import CNN_Block, ConvBlock


####################################################################################################################
################################################## DISCRIMINATEUR ##################################################
####################################################################################################################


class Discriminator(nn.Module):
    def __init__(self, kernel_size=4, stride=2, padding=1, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels*2, features[0], kernel_size, stride, padding, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNN_Block(in_channels, feature, stride=1 if feature ==
                          features[-1] else stride)

            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size, stride=1, padding=padding, padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # X = Correct Satellite Image
        # Y = Correct/Fake Image

        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


####################################################################################################################
#################################################### GENERATEUR ####################################################
####################################################################################################################


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
