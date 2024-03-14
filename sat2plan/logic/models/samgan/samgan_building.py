import torch.nn as nn
import torch
import torch.nn.functional as F

from sat2plan.logic.blocks.blocks import CNN_Block


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
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        out += residual

        return out


class AdaIN(nn.Module):
    def __init__(self, in_channels, style_channels):
        super(AdaIN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * 2)
        )

    def forward(self, content_features, style_features):
        # Calculer les vecteurs de moyenne et de variance à partir des caractéristiques de style
        style_params = self.fc(style_features)
        mean, var = style_params.chunk(2, dim=1)

        # Normaliser les caractéristiques de contenu
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(
            content_features, dim=(2, 3), keepdim=True) + 1e-8
        normalized_content = (content_features - content_mean) / content_std

        # Appliquer le style aux caractéristiques de contenu normalisées
        normalized_content = normalized_content * \
            var.unsqueeze(2).unsqueeze(3) + mean.unsqueeze(2).unsqueeze(3)

        return normalized_content


class ContentEncoder(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=7):
        super(ContentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = SeBlock(64)
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(512, 512) for _ in range(num_residual_blocks)]
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
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, in_channels)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(512, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(512, 512) for _ in range(2)]
        )
        self.adain = AdaIN(512, style_channels)  # Ajouter AdaIN
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(
            64, out_channels, kernel_size=7, stride=1, padding=3)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.tanh = nn.Tanh()

    def forward(self, content_features, style_features):
        out = self.relu(self.norm1(self.conv1(content_features)))
        out = self.residual_blocks(out)
        out = self.adain(out, style_features)  # Utiliser AdaIN
        out = self.upsampling(out)
        out = self.tanh(self.norm2(self.conv2(out)))

        return out


class SAM_GAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=7):
        super(SAM_GAN, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, num_residual_blocks)
        self.style_encoder = StyleEncoder(out_channels)
        # Passer out_channels à Decoder
        self.decoder = Decoder(512, out_channels, out_channels)

    def forward(self, content_img, style_imgs=None):
        if style_imgs is None:
            content_features = self.content_encoder(content_img)
            style_features = self.style_encoder(content_img)
            y_fake = self.decoder(content_features, style_features)

            return y_fake

        content_features = self.content_encoder(content_img)
        style_features = self.style_encoder(style_imgs)
        y_fake = self.decoder(content_features, style_features)

        return y_fake


"""class Discriminator(nn.Module):
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
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)

        return self.model(x)"""


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels*2, 64, kernel_size=4, stride=2, padding=35)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=35)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=35)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=35)
        self.conv5 = nn.Conv2d(
            512, in_channels, kernel_size=4, stride=1, padding=35)

    def forward(self, real_img, generated_img):
        # Concatenate along the channel dimension
        x = torch.cat([real_img, generated_img], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x
