import torch.nn as nn
import torch
import torch.nn.functional as F

from sat2plan.logic.models.blocks.blocks import CNN_Block


class SeBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        out += residual
        return out

class ContentEncoder(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=7):
        super(ContentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = SeBlock(64)
        self.downsampling = nn.Sequential(
            *[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
              nn.InstanceNorm2d(64, affine=True),
              nn.ReLU(inplace=True)]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.se_block(out)
        out = self.downsampling(out)
        out = self.residual_blocks(out)
        return out

class StyleEncoder(nn.Module):
    def __init__(self, in_channels):
        super(StyleEncoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ])
        for _ in range(3):
            self.convs.extend([
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, in_channels)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(2)]
        )
        self.upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
              nn.InstanceNorm2d(64, affine=True),
              nn.ReLU(inplace=True)]
        )
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.residual_blocks(out)
        out = self.upsampling(out)
        out = self.tanh(self.norm2(self.conv2(out)))
        return out

class SAM_GAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, reduction_ratio=16, num_residual_blocks=7):
        super(SAM_GAN, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, num_residual_blocks)
        self.style_encoder = StyleEncoder(out_channels)
        self.decoder = Decoder(64, out_channels)

    def forward(self, content_img, style_imgs):
        content_features = self.content_encoder(content_img)
        style_features = self.style_encoder(style_imgs)
        return self.decoder(content_features, style_features)


"""class SeBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SelfAttention, self).__init__()

        self.se_block = SeBlock(in_channels)

        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Channel attention
        x = self.se_block(x)

        # Project features to query, key, and value with smaller channels
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Compute attention map
        energy = torch.matmul(proj_query.view(batch_size, -1, height * width).permute(0, 2, 1),
                              proj_key.view(batch_size, -1, height * width))
        attention = F.softmax(energy, dim=-1)

        # Compute attention-weighted features
        out = torch.matmul(attention, proj_value.view(batch_size, -1, height * width).permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection
        out = self.gamma * out + x

        return out


class PCIR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PCIR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class CTIR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CTIR, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            PCIR(3, 64, kernel_size=4, padding=2),
            #SelfAttention(64),
            PCIR(64, 128, kernel_size=3, padding=1),
            #SelfAttention(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PCIR(128, 256, kernel_size=4, padding=2),
            #SelfAttention(256),
            PCIR(256, 512, kernel_size=3, padding=1),
            #SelfAttention(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class StyleEncoder(nn.Module):
    def __init__(self, num_maps=512**2):
        super(StyleEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            #SelfAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #SelfAttention(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            #SelfAttention(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #SelfAttention(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upconv_layers = nn.Sequential(
            CTIR(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CTIR(256, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CTIR(64, 3, kernel_size=4, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.upconv_layers(x)
        return x

class SAM_GAN(nn.Module):
    def __init__(self):
        super(SAM_GAN, self).__init__()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.decoder = Decoder()

    def forward(self, x, y):
        content_space = self.content_encoder(x)
        style_space = self.style_encoder(y)
        combined_space = content_space + style_space  # À adapter selon l'architecture exacte
        y_fake = F.interpolate(self.decoder(combined_space), size=(512, 512), mode='bilinear', align_corners=False)
        return y_fake"""

class Discriminator(nn.Module):
    def __init__(self, kernel_size=4, stride=2, padding=2, in_channels=3, features=[64, 64, 64]):
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

"""# Création d'une instance de modèle
model = SAM_GAN()

# Exemple d'utilisation avec des tensors factices
aerial_image = torch.randn(1, 3, 512, 512)  # Adapté à la taille d'entrée 512x512
map_images = torch.randn(1, K, 512, 512)    # K est le nombre de cartes dans le domaine cible
output_map = model(aerial_image, map_images)
print(output_map.shape)  # Vérification de la forme de sortie
"""
