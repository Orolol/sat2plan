import torch
import torch.nn as nn
from sat2plan.logic.blocks.blocks import CNN_Block, ConvBlock


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
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


####################################################################################################################
#################################################### GENERATEUR ####################################################
####################################################################################################################


class Generator(nn.Module):
    def __init__(self,  kernel_size=4, stride=2, padding=1, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size,
                      stride, padding, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )  # 128 X 128

        ##############################################################################
        ################################## ENCODEUR ##################################
        ##############################################################################

        self.down1 = ConvBlock(features, features*2, down=True,
                               act="leaky", use_dropout=False)    # 64 X 64
        self.down2 = ConvBlock(features*2, features*4, down=True,
                               act="leaky", use_dropout=False)  # 32 X 32
        self.down3 = ConvBlock(features*4, features*8, down=True,
                               act="leaky", use_dropout=False)  # 16 X 16
        self.down4 = ConvBlock(features*8, features*8, down=True,
                               act="leaky", use_dropout=False)  # 8 X 8
        self.down5 = ConvBlock(features*8, features*8, down=True,
                               act="leaky", use_dropout=False)  # 4 X 4
        self.down6 = ConvBlock(features*8, features*8, down=True,
                               act="leaky", use_dropout=False)  # 2 X 2

        ##############################################################################
        ################################# BOTTLENECK #################################
        ##############################################################################

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size, stride, padding,
                      padding_mode="reflect"),                      # 1 X 1
            nn.ReLU()
        )

        ##############################################################################
        ################################## DECODEUR ##################################
        ##############################################################################

        self.up1 = ConvBlock(features*8, features*8, down=False,
                             act="relu", use_dropout=True)
        self.up2 = ConvBlock(features*8*2, features*8, down=False,
                             act="relu", use_dropout=True)
        self.up3 = ConvBlock(features*8*2, features*8, down=False,
                             act="relu", use_dropout=True)
        self.up4 = ConvBlock(features*8*2, features*8, down=False,
                             act="relu", use_dropout=False)
        self.up5 = ConvBlock(features*8*2, features*4, down=False,
                             act="relu", use_dropout=False)
        self.up6 = ConvBlock(features*4*2, features*2, down=False,
                             act="relu", use_dropout=False)
        self.up7 = ConvBlock(features*2*2, features, down=False,
                             act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels,
                               kernel_size, stride, padding),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        return self.final_up(torch.cat([up7, d1], 1))
