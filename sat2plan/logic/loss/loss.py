import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad


class AdversarialLoss(nn.Module):
    def __init__(self, mode='bce'):
        super(AdversarialLoss, self).__init__()
        self.mode = mode
        if mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'mse':
            self.loss = nn.MSELoss()
        
    def forward(self, pred, target):
        return self.loss(pred, target)


class ContentLoss(nn.Module):
    def __init__(self, alpha1=1, alpha2=1, alpha3=1):
        super(ContentLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.vgg = models.vgg19(weights='DEFAULT').features[:35].eval()  # Utilise jusqu'à conv5_4
        self.l1_loss = nn.L1Loss()
        
        # Figer les paramètres du VGG
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, y_fake, y):
        # Normalisation des entrées pour VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(y_fake.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(y_fake.device)
        y_fake_norm = (y_fake + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y_norm = (y + 1) / 2
        y_fake_norm = (y_fake_norm - mean) / std
        y_norm = (y_norm - mean) / std
        
        # Extraction des features VGG
        y_fake_features = self.vgg(y_fake_norm)
        y_features = self.vgg(y_norm)
        
        # Perceptual loss (VGG)
        perceptual_loss = self.l1_loss(y_fake_features, y_features) * self.alpha1
        
        # Pixel-level L1 loss
        pixel_loss = self.l1_loss(y_fake, y) * self.alpha2
        
        # Topological consistency loss
        gradient_fake_x = torch.abs(y_fake[:, :, :, 1:] - y_fake[:, :, :, :-1])
        gradient_fake_y = torch.abs(y_fake[:, :, 1:, :] - y_fake[:, :, :-1, :])
        gradient_real_x = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        gradient_real_y = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        
        topo_loss = (self.l1_loss(gradient_fake_x, gradient_real_x) + 
                    self.l1_loss(gradient_fake_y, gradient_real_y)) * self.alpha3
        
        return perceptual_loss + pixel_loss + topo_loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, input):
        batch_size, channel, height, width = input.size()
        features = input.view(batch_size * channel, height * width)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * channel * height * width)

    def forward(self, y_fake, y):
        y_fake_gram = self.gram_matrix(y_fake)
        y_gram = self.gram_matrix(y)
        style_loss = F.l1_loss(y_fake_gram, y_gram)
        return style_loss


class GradientPenalty:
    def __init__(self, batch_size, lambda_gp, device='cuda'):
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.device = device

    def __call__(self, netD, real_samples, fake_samples, condition):
        """Calcule le gradient penalty pour WGAN-GP
        Args:
            netD: le discriminateur
            real_samples: échantillons réels
            fake_samples: échantillons générés
            condition: l'image satellite d'entrée (condition)
        """
        # Génère un nombre aléatoire pour l'interpolation
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        
        # Crée des échantillons interpolés
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
        interpolates.requires_grad_(True)
        
        # Calcule la sortie du discriminateur pour les échantillons interpolés
        d_interpolates = netD(condition, interpolates)

        # Calcule les gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calcule la norme des gradients
        gradients = gradients.view(real_samples.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Calcule la pénalité
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty * self.lambda_gp


def r1_penalty(d_real, real_samples):
    """R1 gradient penalty (Mescheder et al. 2018).

    Pénalise la norme du gradient du discriminateur évalué sur les vrais
    échantillons uniquement. Beaucoup plus stable que WGAN-GP pour un GAN
    conditionnel et compatible avec une loss hinge.

    Args:
        d_real: sortie (logits) du discriminateur sur les vrais échantillons.
        real_samples: le tenseur réel d'entrée (doit avoir requires_grad=True).

    Returns:
        Le scalaire R1 = E[||grad_x D(x)||^2].
    """
    grad = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_samples,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss multi-couches basée sur VGG16 (relu1_2 → relu4_3).

    Adaptée à la traduction sat→carte : les couches basses capturent les
    contours nets (routes, bâti) et les couches hautes la cohérence
    structurelle, là où une L1 pure produit du flou.
    """

    # Indices de fin des blocs relu1_2, relu2_2, relu3_3, relu4_3 dans vgg16.features
    _SLICES = [(0, 4), (4, 9), (9, 16), (16, 23)]

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features.eval()
        self.blocks = nn.ModuleList([
            vgg[start:end] for start, end in self._SLICES
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.weights = weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        # Les images sont dans [-1, 1] (Tanh / Normalize 0.5) -> [0, 1] -> ImageNet
        x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, y_fake, y):
        f = self._normalize(y_fake)
        t = self._normalize(y)
        loss = 0.0
        for block, w in zip(self.blocks, self.weights):
            f = block(f)
            t = block(t)
            loss = loss + w * F.l1_loss(f, t)
        return loss


def feature_matching_loss(feats_fake, feats_real):
    """Feature matching (pix2pixHD) : aligne les activations intermédiaires du
    discriminateur entre faux et vrais. Stabilise l'entraînement et améliore le
    détail bien mieux qu'un signal adversarial seul.

    Les features réelles sont supposées détachées (pas de gradient vers D).
    """
    loss = 0.0
    for ff, fr in zip(feats_fake, feats_real):
        loss = loss + F.l1_loss(ff, fr.detach())
    return loss / max(len(feats_fake), 1)
