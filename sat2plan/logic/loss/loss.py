import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad

# Try to import kornia, if not available use fallback
try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not installed. Using fallback implementation for EdgeLoss.")


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


class EdgeLoss(nn.Module):
    """Loss pour améliorer la netteté des routes et des bords"""
    def __init__(self, weight_sobel=1.0, weight_laplacian=0.5):
        super(EdgeLoss, self).__init__()
        self.weight_sobel = weight_sobel
        self.weight_laplacian = weight_laplacian

        if KORNIA_AVAILABLE:
            # Utiliser kornia si disponible
            self.sobel = kornia.filters.Sobel()
            self.use_kornia = True
        else:
            # Créer les filtres Sobel manuellement
            self.use_kornia = False
            # Filtre Sobel X
            sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # Filtre Sobel Y
            sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Répliquer pour 3 canaux
            self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
            self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

            # Filtre Laplacien pour la détection des bords
            laplacian = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer('laplacian', laplacian.repeat(3, 1, 1, 1))

    def apply_sobel_manual(self, x):
        """Application manuelle du filtre Sobel"""
        # Padding pour maintenir la taille
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # Appliquer les filtres
        edge_x = F.conv2d(x_padded, self.sobel_x.to(x.device), groups=3)
        edge_y = F.conv2d(x_padded, self.sobel_y.to(x.device), groups=3)

        # Magnitude des gradients
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edges

    def apply_laplacian(self, x):
        """Application du filtre Laplacien"""
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        edges = F.conv2d(x_padded, self.laplacian.to(x.device), groups=3)
        return torch.abs(edges)

    def forward(self, y_fake, y_real):
        if self.use_kornia:
            # Version avec kornia
            edges_fake = self.sobel(y_fake)
            edges_real = self.sobel(y_real)
            sobel_loss = F.l1_loss(edges_fake, edges_real) * self.weight_sobel

            # Détection des bords avec Canny
            gray_fake = torch.mean(y_fake, dim=1, keepdim=True)
            gray_real = torch.mean(y_real, dim=1, keepdim=True)
            canny_fake = kornia.filters.canny(gray_fake)[0]
            canny_real = kornia.filters.canny(gray_real)[0]
            canny_loss = F.l1_loss(canny_fake, canny_real) * self.weight_laplacian

            return sobel_loss + canny_loss
        else:
            # Version sans kornia
            # Détection des bords avec Sobel
            edges_fake = self.apply_sobel_manual(y_fake)
            edges_real = self.apply_sobel_manual(y_real)
            sobel_loss = F.l1_loss(edges_fake, edges_real) * self.weight_sobel

            # Détection des bords avec Laplacien (alternative à Canny)
            lap_fake = self.apply_laplacian(y_fake)
            lap_real = self.apply_laplacian(y_real)
            laplacian_loss = F.l1_loss(lap_fake, lap_real) * self.weight_laplacian

            return sobel_loss + laplacian_loss


class ColorSpecificLoss(nn.Module):
    """Loss pour préserver les couleurs spécifiques (ex: jaune pour autoroutes)"""
    def __init__(self, target_colors=None, color_threshold=0.3, weight=1.0):
        super(ColorSpecificLoss, self).__init__()
        self.weight = weight
        self.color_threshold = color_threshold

        # Définir les couleurs cibles importantes (en RGB normalisé [-1, 1])
        if target_colors is None:
            # Jaune pour autoroutes (approximatif en [-1, 1])
            self.target_colors = [
                torch.tensor([0.8, 0.8, -0.5]),  # Jaune
                torch.tensor([0.9, 0.6, -0.8]),  # Orange (routes principales)
                torch.tensor([0.5, 0.5, 0.5]),   # Gris (routes secondaires)
            ]
        else:
            self.target_colors = target_colors

    def detect_color_regions(self, image, target_color):
        """Détecte les régions d'une couleur spécifique"""
        # Calculer la distance dans l'espace colorimétrique
        target_color = target_color.view(1, 3, 1, 1).to(image.device)
        color_distance = torch.norm(image - target_color, dim=1, keepdim=True)

        # Créer un masque pour les pixels proches de la couleur cible
        mask = (color_distance < self.color_threshold).float()
        return mask

    def forward(self, y_fake, y_real):
        total_loss = 0

        for target_color in self.target_colors:
            # Détecter les régions de cette couleur dans l'image réelle
            mask_real = self.detect_color_regions(y_real, target_color)

            # Si il y a des pixels de cette couleur dans l'image réelle
            if mask_real.sum() > 0:
                # Appliquer le masque pour extraire ces régions
                masked_fake = y_fake * mask_real
                masked_real = y_real * mask_real

                # Calculer la loss uniquement sur ces régions
                color_loss = F.l1_loss(masked_fake, masked_real)

                # Pondérer par l'importance de cette couleur
                importance = mask_real.sum() / mask_real.numel()
                total_loss += color_loss * importance

        return total_loss * self.weight


class HighwaySpecificLoss(nn.Module):
    """Loss spécifiquement conçue pour les autoroutes jaunes"""
    def __init__(self, yellow_weight=2.0, width_weight=1.0):
        super(HighwaySpecificLoss, self).__init__()
        self.yellow_weight = yellow_weight
        self.width_weight = width_weight

    def forward(self, y_fake, y_real):
        # Détecter le canal jaune (rouge + vert élevés, bleu faible)
        yellow_mask_real = ((y_real[:, 0] > 0.3) & (y_real[:, 1] > 0.3) & (y_real[:, 2] < 0)).float().unsqueeze(1)

        if yellow_mask_real.sum() > 0:
            # Loss sur les pixels jaunes
            yellow_loss = F.l1_loss(
                y_fake * yellow_mask_real.expand_as(y_fake),
                y_real * yellow_mask_real.expand_as(y_real)
            ) * self.yellow_weight

            # Loss sur la continuité des lignes (dérivées spatiales)
            # Pour s'assurer que les autoroutes sont continues
            grad_x_fake = torch.abs(y_fake[:, :, :, 1:] - y_fake[:, :, :, :-1])
            grad_x_real = torch.abs(y_real[:, :, :, 1:] - y_real[:, :, :, :-1])
            grad_y_fake = torch.abs(y_fake[:, :, 1:, :] - y_fake[:, :, :-1, :])
            grad_y_real = torch.abs(y_real[:, :, 1:, :] - y_real[:, :, :-1, :])

            # Appliquer le masque jaune sur les gradients
            mask_x = yellow_mask_real[:, :, :, 1:]
            mask_y = yellow_mask_real[:, :, 1:, :]

            continuity_loss = (
                F.l1_loss(grad_x_fake * mask_x, grad_x_real * mask_x) +
                F.l1_loss(grad_y_fake * mask_y, grad_y_real * mask_y)
            ) * self.width_weight

            return yellow_loss + continuity_loss
        else:
            return torch.tensor(0.0).to(y_fake.device)


"""# Exemple d'utilisation :
alpha1 = 1.0
alpha2 = 1.0
alpha3 = 1.0

adversarial_loss = AdversarialLoss()
content_loss = ContentLoss(alpha1, alpha2, alpha3)
style_loss = StyleLoss()
edge_loss = EdgeLoss()
color_loss = ColorSpecificLoss()
highway_loss = HighwaySpecificLoss()

# Exemple de calcul de chaque perte
adv_loss = adversarial_loss(y_fake, y)
content_loss_value = content_loss(y_fake, y)
style_loss_value = style_loss(y_fake, y)
edge_loss_value = edge_loss(y_fake, y)
color_loss_value = color_loss(y_fake, y)
highway_loss_value = highway_loss(y_fake, y)"""
