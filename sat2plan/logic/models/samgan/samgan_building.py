import torch.nn as nn
import torch
import torch.nn.functional as F

from sat2plan.logic.models.blocks.blocks import CNN_Block

class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class StyleEncoder(nn.Module):
    def __init__(self, num_maps=512**2):
        super(StyleEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
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
        # À compléter avec l'initialisation des autres composants du modèle

    def forward(self, x, y):
        content_space = self.content_encoder(x)
        style_space = self.style_encoder(y)
        combined_space = content_space + style_space  # À adapter selon l'architecture exacte
        y_fake = self.decoder(combined_space)
        return y_fake

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

        x = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False)

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
