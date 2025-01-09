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


"""# Exemple d'utilisation :
alpha1 = 1.0
alpha2 = 1.0
alpha3 = 1.0

adversarial_loss = AdversarialLoss()
content_loss = ContentLoss(alpha1, alpha2, alpha3)
style_loss = StyleLoss()

# Exemple de calcul de chaque perte

adv_loss = adversarial_loss(y_fake, y)

content_loss_value = content_loss(y_fake, y)

style_loss_value = style_loss(y_fake, y)"""
