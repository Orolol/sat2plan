"""Diffusion gaussienne pour traduction conditionnelle sat→carte (style Palette).

Choix de conception :
- **v-prediction** (Salimans & Ho, 2022) : cible plus stable que la prédiction de
  bruit, surtout aux pas de temps extrêmes.
- **Min-SNR-γ** (Hang et al., 2023) : pondération de la loss qui équilibre les
  contributions des différents pas de temps → convergence nettement plus rapide.
- **Conditionnement par concaténation** : l'image satellite est empilée sur le
  canal de la cible bruitée *à l'extérieur* (dans le trainer / sampler), ce qui
  préserve l'alignement pixel-à-pixel — l'atout d'une tâche appariée.
"""

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 2e-2, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


def _extract(a, t, shape):
    """Indexe le buffer 1D `a` au pas `t` et reshape pour broadcaster sur `shape`."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000, schedule="cosine", prediction_type="v",
                 min_snr_gamma=5.0, device=None):
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.device = device if device is not None else torch.device("cpu")

        betas = (cosine_beta_schedule(timesteps) if schedule == "cosine"
                 else linear_beta_schedule(timesteps)).to(self.device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = acp
        self.alphas_cumprod_prev = acp_prev
        self.sqrt_acp = acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - acp).sqrt()

        # Pondération Min-SNR-γ selon la paramétrisation.
        snr = acp / (1.0 - acp)
        clamped = snr.clamp(max=min_snr_gamma)
        if prediction_type == "v":
            self.loss_weight = clamped / (snr + 1.0)
        elif prediction_type == "eps":
            self.loss_weight = clamped / snr
        else:  # x0
            self.loss_weight = clamped

    # ---- conversions de paramétrisation ----
    def predict_v(self, x_start, t, noise):
        return (_extract(self.sqrt_acp, t, x_start.shape) * noise
                - _extract(self.sqrt_one_minus_acp, t, x_start.shape) * x_start)

    def predict_start_from_v(self, x_t, t, v):
        return (_extract(self.sqrt_acp, t, x_t.shape) * x_t
                - _extract(self.sqrt_one_minus_acp, t, x_t.shape) * v)

    def predict_start_from_eps(self, x_t, t, eps):
        return ((x_t - _extract(self.sqrt_one_minus_acp, t, x_t.shape) * eps)
                / _extract(self.sqrt_acp, t, x_t.shape))

    def q_sample(self, x_start, t, noise):
        return (_extract(self.sqrt_acp, t, x_start.shape) * x_start
                + _extract(self.sqrt_one_minus_acp, t, x_start.shape) * noise)

    def model_target(self, x_start, t, noise):
        if self.prediction_type == "v":
            return self.predict_v(x_start, t, noise)
        elif self.prediction_type == "eps":
            return noise
        return x_start

    def x0_from_pred(self, x_t, t, pred):
        if self.prediction_type == "v":
            return self.predict_start_from_v(x_t, t, pred)
        elif self.prediction_type == "eps":
            return self.predict_start_from_eps(x_t, t, pred)
        return pred

    def p_losses(self, model, x_start, t, cond, noise=None):
        """Loss d'entraînement. `cond` est concaténé à la cible bruitée."""
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = model(torch.cat([x_noisy, cond], dim=1), t)
        target = self.model_target(x_start, t, noise)

        loss = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        weight = self.loss_weight.gather(0, t)
        return (loss * weight).mean()

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, device, steps=50, eta=0.0):
        """Échantillonnage DDIM (déterministe par défaut, eta=0)."""
        # Sous-séquence de pas de temps, décroissante de T-1 à 0.
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()
        img = torch.randn(shape, device=device)

        for i in range(steps):
            t = times[i]
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred = model(torch.cat([img, cond], dim=1), t_batch)

            x0 = self.x0_from_pred(img, t_batch, pred).clamp(-1.0, 1.0)

            acp_t = self.alphas_cumprod[t]
            acp_next = self.alphas_cumprod[times[i + 1]] if i < steps - 1 else torch.tensor(1.0, device=device)

            # Bruit cohérent recalculé depuis x0 (robuste au clamp).
            eps = (img - acp_t.sqrt() * x0) / (1.0 - acp_t).sqrt().clamp(min=1e-8)
            sigma = eta * (((1 - acp_next) / (1 - acp_t)).clamp(min=0).sqrt()
                           * (1 - acp_t / acp_next).clamp(min=0).sqrt())
            dir_xt = (1 - acp_next - sigma ** 2).clamp(min=0).sqrt() * eps
            img = acp_next.sqrt() * x0 + dir_xt
            if eta > 0 and i < steps - 1:
                img = img + sigma * torch.randn_like(img)

        return img.clamp(-1.0, 1.0)
