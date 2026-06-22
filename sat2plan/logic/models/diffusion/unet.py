"""UNet conditionnel pour diffusion sat→carte (style Palette / ADM).

Conditionnement par **concaténation** : l'entrée est `cat([x_t, cond], dim=1)`
(6 canaux), ce qui préserve l'alignement spatial pixel-à-pixel — crucial pour une
tâche appariée, contrairement à un conditionnement par cross-attention global.

Briques : ResnetBlock (GroupNorm + SiLU + modulation FiLM par le temps),
self-attention multi-tête aux basses résolutions, skips ADM-style.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def Normalize(channels):
    # Plus grand nombre de groupes <= 32 qui divise `channels` (robuste à toute largeur).
    num_groups = next((g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0), 1)
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)


class ResnetBlock(nn.Module):
    """Bloc résiduel avec modulation FiLM (scale-shift) du temps."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = Normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention spatiale multi-tête (résiduelle)."""

    def __init__(self, channels, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            return t.reshape(b, self.heads, c // self.heads, h * w).transpose(2, 3)

        q, k, v = map(reshape_heads, (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, out_channels=3, base=64,
                 ch_mult=(1, 2, 2, 4, 4), num_res_blocks=2, attn_resolutions=(32, 16),
                 time_dim=256, image_size=256, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels + cond_channels, base, 3, padding=1)

        num_levels = len(ch_mult)
        skip_channels = [base]
        cur_ch = base
        cur_res = image_size

        # ---------- Encoder ----------
        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            out_ch = base * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleDict({
                    "res": ResnetBlock(cur_ch, out_ch, time_dim, dropout),
                    "attn": AttnBlock(out_ch) if cur_res in attn_resolutions else nn.Identity(),
                }))
                cur_ch = out_ch
                skip_channels.append(cur_ch)
            if level != num_levels - 1:
                self.down_blocks.append(nn.ModuleDict({"downsample": Downsample(cur_ch)}))
                skip_channels.append(cur_ch)
                cur_res //= 2

        # ---------- Middle ----------
        self.mid_res1 = ResnetBlock(cur_ch, cur_ch, time_dim, dropout)
        self.mid_attn = AttnBlock(cur_ch)
        self.mid_res2 = ResnetBlock(cur_ch, cur_ch, time_dim, dropout)

        # ---------- Decoder ----------
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(nn.ModuleDict({
                    "res": ResnetBlock(cur_ch + skip_channels.pop(), out_ch, time_dim, dropout),
                    "attn": AttnBlock(out_ch) if cur_res in attn_resolutions else nn.Identity(),
                }))
                cur_ch = out_ch
            if level != 0:
                self.up_blocks.append(nn.ModuleDict({"upsample": Upsample(cur_ch)}))
                cur_res *= 2

        self.out_norm = Normalize(cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, out_channels, 3, padding=1)
        # Sortie initialisée à zéro : départ d'entraînement stable (prédit 0).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        t = self.time_mlp(t)
        h = self.in_conv(x)
        hs = [h]

        for block in self.down_blocks:
            if "downsample" in block:
                h = block["downsample"](h)
            else:
                h = block["res"](h, t)
                h = block["attn"](h)
            hs.append(h)

        h = self.mid_res1(h, t)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t)

        for block in self.up_blocks:
            if "upsample" in block:
                h = block["upsample"](h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = block["res"](h, t)
                h = block["attn"](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
