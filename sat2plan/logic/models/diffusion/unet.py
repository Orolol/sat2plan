import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.up = up
        
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t, skip=None):
        if self.up and skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))

        # Efficient attention using Flash Attention or memory-efficient attention
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(out.shape[0], -1, q.shape[-1] * h)
        return self.to_out(out)

class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.attention = CrossAttention(channels, context_dim, heads=num_heads)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x = self.norm(x)
        x = x.reshape(b, c, -1).transpose(1, 2)
        x = self.attention(x, context)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, context_dim=768):
        super().__init__()
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder for conditioning image (satellite)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, context_dim, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Downsampling
        self.down1 = Block(64, 128, time_dim)
        self.sa1 = SpatialTransformer(128, context_dim)
        self.down2 = Block(128, 256, time_dim)
        self.sa2 = SpatialTransformer(256, context_dim)
        self.down3 = Block(256, 512, time_dim)
        self.sa3 = SpatialTransformer(512, context_dim)

        # Bottleneck
        self.bot1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot_sa = SpatialTransformer(512, context_dim)

        # Upsampling
        self.up1 = Block(512, 256, time_dim, up=True)
        self.sa4 = SpatialTransformer(256, context_dim)
        self.up2 = Block(256, 128, time_dim, up=True)
        self.sa5 = SpatialTransformer(128, context_dim)
        self.up3 = Block(128, 64, time_dim, up=True)
        self.sa6 = SpatialTransformer(64, context_dim)
        
        # Final conv
        self.final_conv = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x, t, context):
        # Encode conditioning image
        context_features = self.context_encoder(context)
        b, c, h, w = context_features.shape
        context_features = context_features.reshape(b, c, -1).transpose(1, 2)  # (b, h*w, c)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Time embedding
        t = self.time_mlp(t)
        
        # Unet
        # Downsample
        d1 = self.sa1(self.down1(x, t), context_features)
        d2 = self.sa2(self.down2(d1, t), context_features)
        d3 = self.sa3(self.down3(d2, t), context_features)
        
        # Bottleneck
        bot = F.relu(self.bot1(d3))
        bot = F.relu(self.bot2(bot))
        bot = self.bot_sa(F.relu(self.bot3(bot)), context_features)
        
        # Upsample with skip connections
        up1 = self.sa4(self.up1(bot, t, d3), context_features)
        up2 = self.sa5(self.up2(up1, t, d2), context_features)
        up3 = self.sa6(self.up3(up2, t, d1), context_features)
        
        return self.final_conv(up3) 