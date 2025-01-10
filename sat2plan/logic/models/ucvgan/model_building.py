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
        
        # Dynamic feature extraction blocks
        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(32, 64),
                nn.AvgPool2d(2),
                ResidualBlock(64, 64)
            ),
            nn.Sequential(
                ResidualBlock(64, 128),
                nn.AvgPool2d(2),
                ResidualBlock(128, 128)
            ),
            nn.Sequential(
                ResidualBlock(128, 256),
                nn.AvgPool2d(2),
                ResidualBlock(256, 256)
            )
        ])
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
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
        
        # Apply feature extraction blocks
        for block in self.features:
            x = block(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        return self.final(x)


####################################################################################################################
#################################################### GENERATEUR ####################################################
####################################################################################################################


class Generator(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, in_channels=3, features=64, input_size=256):
        super().__init__()
        
        # Calculate number of encoder blocks needed
        self.n_blocks = max(5, (input_size // 256) + 4)  # 5 blocks for 256x256, 6 for 512x512, etc.
        final_size = input_size // (2 ** self.n_blocks)  # Size after all encoder blocks
        
        # Initial downsampling avec normalisation
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size, stride, padding, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder blocks with increased feature sizes
        self.encoder = nn.ModuleList()
        current_features = features
        for i in range(self.n_blocks):
            out_features = min(features * 8, current_features * 2)  # Cap at features * 8
            self.encoder.append(
                nn.ModuleDict({
                    'conv': UVCCNNlock(current_features, out_features, down=True),
                    'scale': DownsamplingBlock(out_features, out_features)
                })
            )
            current_features = out_features

        # Bottleneck with increased attention heads and blocks
        self.bottleneck = PixelwiseViT(
            features * 8,  # input features
            16,           # n_heads
            12,           # n_blocks
            2048,         # ffn_features
            features * 8, # embed_features
            image_shape=(features * 8, final_size, final_size),  # shape après l'encodeur
            rezero=True,
            dropout=0.1
        )

        # Decoder blocks with skip connections and increased features
        self.decoder = nn.ModuleList()
        for i in range(self.n_blocks):
            if i < 3:  # Premiers blocs
                in_features = features * 16
                out_features = features * 8
            elif i < 4:  # Quatrième bloc
                in_features = features * 12
                out_features = features * 4
            else:  # Derniers blocs
                in_features = features * 6
                out_features = features * 2
            
            self.decoder.append(
                nn.ModuleDict({
                    'scale': UpsamplingBlock(in_features, in_features),
                    'conv': UVCCNNlock(in_features, out_features, down=False)
                })
            )

        # Final upsampling avec plus de couches
        self.final_up = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features // 2, in_channels, kernel_size=1, stride=1, padding=0),
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
            # Ensure skip connection has same spatial dimensions
            if current.shape[-2:] != skip_connection.shape[-2:]:
                skip_connection = nn.functional.interpolate(
                    skip_connection, 
                    size=current.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            current = torch.cat([current, skip_connection], dim=1)
            current = block['conv'](block['scale'](current))
        
        # Final upsampling
        result = self.final_up(current)
        
        return result
