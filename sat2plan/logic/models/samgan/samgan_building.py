import torch.nn as nn
import torch
import torch.nn.functional as F

from sat2plan.logic.models.blocks.blocks import CNN_Block


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Project features to query, key, and value
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, height * width)

        # Compute attention-weighted features
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
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
        return y_fake

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
