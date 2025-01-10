import torch
import torch.nn as nn
from sat2plan.logic.blocks.blocks import CNN_Block, UVCCNNlock, PixelwiseViT, DownsamplingBlock, UpsamplingBlock


####################################################################################################################
################################################## DISCRIMINATEUR ##################################################
####################################################################################################################


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="reflect", bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        
    def forward(self, x):
        residual = x
        out = self.lrelu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.lrelu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, 32, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.final(x)


####################################################################################################################
#################################################### GENERATEUR ####################################################
####################################################################################################################


class Generator(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, in_channels=3, features=96):
        super().__init__()
        
        # Initial downsampling avec normalisation
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(features, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            # nn.InstanceNorm2d(features, affine=True),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder blocks with increased feature sizes
        self.encoder = nn.ModuleList([
            # Block 1: (B, 64, 256, 256) -> (B, 128, 128, 128)
            nn.ModuleDict({
                'conv': UVCCNNlock(features, features*2, down=True),
                'scale': DownsamplingBlock(features*2, features*2)
            }),
            # Block 2: (B, 128, 128, 128) -> (B, 256, 64, 64)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*2, features*4, down=True),
                'scale': DownsamplingBlock(features*4, features*4)
            }),
            # Block 3: (B, 256, 64, 64) -> (B, 512, 32, 32)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*4, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            }),
            # Block 4: (B, 512, 32, 32) -> (B, 512, 16, 16)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*8, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            }),
            # Block 5: (B, 512, 16, 16) -> (B, 512, 8, 8)
            nn.ModuleDict({
                'conv': UVCCNNlock(features*8, features*8, down=True),
                'scale': DownsamplingBlock(features*8, features*8)
            })
        ])

        # Bottleneck with increased attention heads and blocks
        self.bottleneck = PixelwiseViT(
            features * 8, 16, 12, 2048,  # Plus de têtes d'attention et de blocs
            features * 8,
            image_shape=(features * 8, 8, 8),
            rezero=True,
            dropout=0.1  # Ajout de dropout pour régularisation
        )

        # Decoder blocks with skip connections and increased features
        self.decoder = nn.ModuleList([
            # Block 1: (B, 1024, 8, 8) -> (B, 512, 16, 16)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 2: (B, 1024, 16, 16) -> (B, 512, 32, 32)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 3: (B, 1024, 32, 32) -> (B, 512, 64, 64)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*16, features*16),
                'conv': UVCCNNlock(features*16, features*8, down=False)
            }),
            # Block 4: (B, 768, 64, 64) -> (B, 256, 128, 128)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*12, features*12),
                'conv': UVCCNNlock(features*12, features*4, down=False)
            }),
            # Block 5: (B, 384, 128, 128) -> (B, 128, 256, 256)
            nn.ModuleDict({
                'scale': UpsamplingBlock(features*6, features*6),
                'conv': UVCCNNlock(features*6, features*2, down=False)
            })
        ])

        # Final upsampling avec plus de couches
        self.final_up = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(features // 2, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(features // 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        # Initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        # Initial downsampling
        d1 = self.initial_down(x)
        
        # Encoder with memory optimization
        encoder_features = []
        current = d1
        for block in self.encoder:
            current = block['scale'](block['conv'](current))
            encoder_features.append(current)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoder_features[-1])
        
        # Decoder with optimized skip connections
        current = bottleneck
        for idx, block in enumerate(self.decoder):
            skip_connection = encoder_features[-(idx+1)]
            current = torch.cat([current, skip_connection], dim=1)
            current = block['conv'](block['scale'](current))
        
        # Final upsampling
        result = self.final_up(current)
        
        return result
