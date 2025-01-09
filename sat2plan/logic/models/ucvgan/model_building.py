import torch
import torch.nn as nn
from sat2plan.logic.blocks.blocks import CNN_Block, UVCCNNlock, PixelwiseViT, DownsamplingBlock, UpsamplingBlock


####################################################################################################################
################################################## DISCRIMINATEUR ##################################################
####################################################################################################################


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="reflect")
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.in2 = nn.InstanceNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = nn.LeakyReLU(0.2, inplace=True)(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += self.shortcut(residual)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks with increasing features
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        # Concatenate input and condition
        x = torch.cat([x, y], dim=1)
        
        # Forward pass through network
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        
        return x


####################################################################################################################
#################################################### GENERATEUR ####################################################
####################################################################################################################


class Generator(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, in_channels=3, features=48):
        super().__init__()
        
        # Initial downsampling: (B, 3, 256, 256) -> (B, 48, 256, 256)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            nn.LeakyReLU(0.2, inplace=True)
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
            features * 8, 8, 8, 1536,  # Réduit encore la taille du bottleneck
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
