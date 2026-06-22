"""Métriques de qualité d'image pour la traduction sat→carte.

Regroupe FID, LPIPS et SSIM (torchmetrics) derrière une interface robuste :
chaque métrique est optionnelle et son échec d'initialisation (ex. poids
indisponibles hors-ligne) est dégradé en avertissement, sans interrompre
l'entraînement.

Convention d'entrée : images dans [-1, 1] (sortie Tanh du générateur). La
conversion vers [0, 1] attendue par les métriques est faite ici.
"""

import torch
import torch.nn as nn


def _to_unit_range(x):
    """[-1, 1] -> [0, 1], clampé, en float32."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0).float()


class ImageQualityMetrics(nn.Module):
    """Accumulateur de métriques de qualité (LPIPS, SSIM, FID).

    Usage :
        metrics = ImageQualityMetrics(device)
        metrics.reset()
        for fake, real in ...:        # tenseurs dans [-1, 1]
            metrics.update(fake, real)
        results = metrics.compute()   # {'lpips': ..., 'ssim': ..., 'fid': ...}
    """

    def __init__(self, device, enable_fid=True, enable_lpips=True, enable_ssim=True):
        super().__init__()
        self.device = device
        self.ssim = None
        self.lpips = None
        self.fid = None

        if enable_ssim:
            try:
                from torchmetrics.image import StructuralSimilarityIndexMeasure
                self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            except Exception as e:
                print(f"⚠️  SSIM indisponible: {e}")

        if enable_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                # normalize=True -> les entrées sont attendues dans [0, 1]
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
            except Exception as e:
                print(f"⚠️  LPIPS indisponible: {e}")

        if enable_fid:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                # normalize=True -> entrées float dans [0, 1]
                self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            except Exception as e:
                print(f"⚠️  FID indisponible: {e}")

        self.to(device)
        active = [n for n, m in (('lpips', self.lpips), ('ssim', self.ssim), ('fid', self.fid)) if m is not None]
        print(f"📐 Métriques actives: {active if active else 'aucune'}")

    @property
    def has_lpips(self):
        return self.lpips is not None

    def reset(self):
        for m in (self.ssim, self.lpips, self.fid):
            if m is not None:
                m.reset()

    @torch.no_grad()
    def update(self, fake, real):
        """fake/real : tenseurs (B, 3, H, W) dans [-1, 1]."""
        fake_u = _to_unit_range(fake)
        real_u = _to_unit_range(real)

        if self.ssim is not None:
            self.ssim.update(fake_u, real_u)
        if self.lpips is not None:
            self.lpips.update(fake_u, real_u)
        if self.fid is not None:
            self.fid.update(real_u, real=True)
            self.fid.update(fake_u, real=False)

    @torch.no_grad()
    def compute(self):
        """Retourne un dict des métriques disponibles (valeurs Python)."""
        results = {}
        if self.ssim is not None:
            results['ssim'] = float(self.ssim.compute())
        if self.lpips is not None:
            results['lpips'] = float(self.lpips.compute())
        if self.fid is not None:
            try:
                results['fid'] = float(self.fid.compute())
            except Exception as e:
                # FID lève si trop peu d'échantillons accumulés
                print(f"⚠️  FID non calculable: {e}")
        return results
