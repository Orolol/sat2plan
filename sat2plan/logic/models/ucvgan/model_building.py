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
            nn.LeakyReLU(0.2, inplace=False)
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
    def __init__(self, kernel_size=3, stride=1, padding=1, in_channels=3, features=48):
        super().__init__()
        
        # Initial downsampling: (B, 3, 256, 256) -> (B, 48, 256, 256)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            nn.LeakyReLU(0.2, inplace=False)
        )

        # Encoder blocks with reduced feature sizes
        self.encoder = nn.ModuleList([
            # Block 1: (B, 48, 256, 256) -> (B, 96, 128, 128)
            nn.ModuleDict({
                'conv': UVCCNNlock(features, features*2, down=True),
                'scale': DownsamplingBlock(features*2, features*2)
            }),
            # Block 2: (B, 96, 128, 128) -> (B, 192, 64, 64)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*2, features*4, down=True),
                'scale': DownsamplingBlock(features*4, features*4)
            }),
            # Block 3: (B, 192, 64, 64) -> (B, 384, 32, 32)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*4, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            }),
            # Block 4: (B, 384, 32, 32) -> (B, 384, 16, 16)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*8, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            }),
            # Block 5: (B, 384, 16, 16) -> (B, 384, 8, 8)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*8, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            })
        ])

        # Bottleneck: (B, 384, 8, 8) -> (B, 384, 8, 8)
        self.bottleneck = PixelwiseViT(
            features * 8, 8, 8, 768,  # Réduit encore la taille du bottleneck
            features * 8,
            image_shape=(features * 8, 8, 8),
            rezero=True
        )

        # Decoder blocks
        self.decoder = nn.ModuleList([
            # Block 1: (B, 768, 8, 8) -> (B, 384, 16, 16)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 2: (B, 768, 16, 16) -> (B, 384, 32, 32)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 3: (B, 768, 32, 32) -> (B, 384, 64, 64)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 4: (B, 576, 64, 64) -> (B, 192, 128, 128)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*12, features*12),
                'conv': UVCCNNlock(features*12, features*4, down=False)
            }),
            # Block 5: (B, 288, 128, 128) -> (B, 96, 256, 256)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*6, features*6),
                'conv': UVCCNNlock(features*6, features*2, down=False)
            })
        ])

        # Final upsampling: (B, 96, 256, 256) -> (B, 3, 256, 256)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # Initial downsampling
        d1 = self.initial_down(x)
        
        # Encoder with memory optimization
        encoder_features = []
        current = d1
        for block in self.encoder:
            current = block['scale'](block['conv'](current))
            encoder_features.append(current.detach())
        
        # Bottleneck
        bottleneck = self.bottleneck(encoder_features[-1])
        
        # Decoder with optimized skip connections
        current = bottleneck
        for idx, block in enumerate(self.decoder):
            skip_connection = encoder_features[-(idx+1)]
            current = torch.cat([current, skip_connection], dim=1)
            current = block['conv'](block['scale'](current))
            del skip_connection
        
        # Final upsampling
        result = self.final_up(current)
        
        return result
