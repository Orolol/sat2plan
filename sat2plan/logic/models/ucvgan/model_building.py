import torch
import torch.nn as nn
from sat2plan.logic.blocks.blocks import CNN_Block, UVCCNNlock, PixelwiseViT, DownsamplingBlock, UpsamplingBlock


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
    def __init__(self,  kernel_size=3, stride=1, padding=1, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size,
                      stride, padding, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )  # 256 X 256

        ##############################################################################
        ################################## ENCODEUR ##################################
        ##############################################################################

        self.down1 = UVCCNNlock(features, features*2, down=True)    # 64 X 64
        self.dscale1 = DownsamplingBlock(features*2, features*2)
        self.down2 = UVCCNNlock(features*2, features*4, down=True)  # 32 X 32
        self.dscale2 = DownsamplingBlock(features*4, features*4)
        self.down3 = UVCCNNlock(features*4, features*8, down=True)  # 16 X 16
        self.dscale3 = DownsamplingBlock(features*8, features*8)
        self.down4 = UVCCNNlock(features*8, features*8, down=True)  # 16 X 16
        self.dscale4 = DownsamplingBlock(features*8, features*8)
        self.down5 = UVCCNNlock(features*8, features*8, down=True)  # 16 X 16
        self.dscale5 = DownsamplingBlock(features*8, features*8)

        ##############################################################################
        ################################# BOTTLENECK #################################
        ##############################################################################

        self.bottleneck = PixelwiseViT(
            features * 8, 8, 8, 1536, features * 8,
            image_shape=(features * 8, 16, 16),
            rezero=True
        )

        ##############################################################################
        ################################## DECODEUR ##################################
        ##############################################################################

        self.uscale3 = UpsamplingBlock(features*16, features*16)
        self.up3 = UVCCNNlock(features*16, features*8, down=False)   # 16 * 16
        self.uscale4 = UpsamplingBlock(features*16, features*16)
        self.up4 = UVCCNNlock(features*16, features*8, down=False)   # 16 * 16
        self.uscale5 = UpsamplingBlock(features*16, features*16)
        self.up5 = UVCCNNlock(features*16, features*8, down=False)
        self.uscale6 = UpsamplingBlock(features*12, features*12)
        self.up6 = UVCCNNlock(features*12, features*4, down=False)
        self.uscale7 = UpsamplingBlock(features*6, features*6)
        self.up7 = UVCCNNlock(features*6, features*2, down=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels,
                               kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):

        debug_mode = False

        d1 = self.initial_down(x)
        print("D1 shape", d1.shape) if debug_mode else None
        d2 = self.dscale1(self.down1(d1))
        print("D2 shape", d2.shape) if debug_mode else None
        d3 = self.dscale2(self.down2(d2))
        print("D3 shape", d3.shape) if debug_mode else None
        d4 = self.dscale3(self.down3(d3))
        print("D4 shape", d4.shape) if debug_mode else None
        d5 = self.dscale4(self.down4(d4))
        print("D5 shape", d5.shape) if debug_mode else None
        d6 = self.dscale5(self.down5(d5))
        print("D6 shape", d6.shape) if debug_mode else None

        bottleneck = self.bottleneck(d6)
        # 512 X 16 X 16
        print("Bottleneck shape", bottleneck.shape) if debug_mode else None

        up1 = self.up3(self.uscale3(torch.cat([bottleneck, d6], 1)))
        print("Up1 shape", up1.shape) if debug_mode else None
        up2 = self.up4(self.uscale4(torch.cat([up1, d5], 1)))  # 512 X 64 X 64
        print("Up2shape", up2.shape) if debug_mode else None
        up3 = self.up5(self.uscale5(torch.cat([up2, d4], 1)))
        print("Up3 shape", up3.shape) if debug_mode else None
        up4 = self.up6(self.uscale6(torch.cat([up3, d3], 1)))
        print("Up4 shape", up4.shape) if debug_mode else None
        up5 = self.up7(self.uscale7(torch.cat([up4, d2], 1)))
        print("Up5 shape", up5.shape) if debug_mode else None

        result = self.final_up(up5)

        print("Result shape", result.shape) if debug_mode else None

        return result
