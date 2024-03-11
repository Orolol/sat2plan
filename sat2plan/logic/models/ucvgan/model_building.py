import torch
import torch.nn as nn
from sat2plan.logic.models.blocks.blocks import CNN_Block, ConvBlock, PixelwiseViT


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
        # print("X", x.shape)
        # print("Y", y.shape)
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
                               act="leaky", use_dropout=False)  # 16 X 16
        self.down5 = ConvBlock(features*8, features*8, down=True,
                               act="leaky", use_dropout=False)  # 16 X 16
        # self.down4 = ConvBlock(features*8, features*8, stride=1, padding=1, kernel_size=3, down=True,
        #                        act="leaky", use_dropout=False)  # 16 X 16

        ##############################################################################
        ################################# BOTTLENECK #################################
        ##############################################################################
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(features*8, features*8, kernel_size, stride, padding,
        #               padding_mode="reflect"),                      # 1 X 1
        #     nn.ReLU()
        # )

        self.bottleneck = PixelwiseViT(
            features * 8, 8, 8, 1536, features * 8,
            image_shape=(features * 8, 8, 8),
            rezero=True
        )

        ##############################################################################
        ################################## DECODEUR ##################################
        ##############################################################################

        self.up3 = ConvBlock(features * 8, features*8, down=False,
                             act="relu", use_dropout=False)   # 16 * 16
        self.up4 = ConvBlock(features*16, features*8, down=False,
                             act="relu", use_dropout=False)   # 16 * 16
        self.up5 = ConvBlock(features*16, features*8, down=False,
                             act="relu", use_dropout=False)
        self.up6 = ConvBlock(features*12, features*4, down=False,
                             act="relu", use_dropout=False)
        self.up7 = ConvBlock(features*6, features*2, down=False,
                             act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        # print(1)
        d2 = self.down1(d1)
        # print(2)
        d3 = self.down2(d2)
        # print(3)
        d4 = self.down3(d3)
        # print(4)
        d5 = self.down4(d4)
        # print(5)
        # print("D5", d5.shape)
        d6 = self.down5(d5)
        # print("D6", d6.shape)
        bottleneck = self.bottleneck(d6)
        # print(6)
        # up4 = self.up4(torch.cat([up1, d5], 1))
        up1 = self.up3(bottleneck)
        # print("UP1", up1.shape)
        # print("d5", d5.shape)
        # print("D4", d4.shape)
        # print("d3", d3.shape)
        # print("d2", d2.shape)
        # print("d1", d1.shape)
        up3 = self.up4(torch.cat([up1, d5], 1))
        # print("UP3", up3.shape)
        up4 = self.up5(torch.cat([up3, d4], 1))
        # print("UP4", up4.shape)
        up5 = self.up6(torch.cat([up4, d3], 1))
        # print("UP5", up5.shape)

        up6 = self.up7(torch.cat([up5, d2], 1))
        # # print("UP6", up6.shape)

        result = self.final_up(up6)
        # print("FINAL G", result.shape)
        return result
