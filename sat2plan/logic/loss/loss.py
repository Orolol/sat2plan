import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, y_fake, y):
        adversarial_loss = torch.mean(torch.log(y) + torch.log(1 - y_fake))
        return adversarial_loss

class ContentLoss(nn.Module):
    def __init__(self, alpha1=1, alpha2=1, alpha3=1):
        super(ContentLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.vgg = models.vgg19(weights='DEFAULT').features.eval()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_fake, y):
        y_fake_features = self.vgg(y_fake)
        y_features = self.vgg(y)

        vgg_loss = 0
        pixel_level_loss = self.l1_loss(y_fake, y)

        for y_fake_feat, y_feat in zip(y_fake_features, y_features):
            vgg_loss += self.l1_loss(y_fake_feat, y_feat)

        vgg_loss *= self.alpha1

        # Calcul de la topological consistency loss (L_top)
        gradient_generated_x = torch.abs(y_fake[:, :, :-1, :] - y_fake[:, :, 1:, :])
        gradient_generated_y = torch.abs(y_fake[:, :, :, :-1] - y_fake[:, :, :, 1:])
        gradient_real_x = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        gradient_real_y = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])

        topological_loss = self.l1_loss(gradient_generated_x, gradient_real_x) + self.l1_loss(gradient_generated_y, gradient_real_y)
        topological_loss *= self.alpha3

        total_content_loss = vgg_loss + self.alpha2 * pixel_level_loss + topological_loss
        return total_content_loss

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
