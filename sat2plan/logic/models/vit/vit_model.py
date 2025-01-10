import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # (B, C, H, W) -> (B, embed_dim, H', W')
        x = self.proj(x)
        
        # (B, embed_dim, H', W') -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Layer normalization
        x = self.norm(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        
        # Split heads and generate q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use torch's optimized scaled dot product attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        
        # Reshape back
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding for patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # Shape: [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, num_patches + 1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 1:]  # Remove class token

class ViTDecoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, out_channels=3, embed_dim=768,
                 num_heads=12, num_layers=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Add position embedding for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, patch_size * patch_size * out_channels)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x)

        # Reshape into image patches
        B = x.shape[0]
        x = x.view(B, self.n_patches, self.patch_size, self.patch_size, self.out_channels)
        h = w = int(math.sqrt(self.n_patches))
        x = x.view(B, h, w, self.patch_size, self.patch_size, self.out_channels)
        
        # Merge patches
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.img_size, self.img_size)
        return torch.tanh(x)

class ViTSat2Plan(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, out_channels=3,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_channels, embed_dim,
                                num_heads, num_layers, mlp_ratio, dropout)
        self.decoder = ViTDecoder(img_size, patch_size, out_channels, embed_dim,
                                num_heads, num_layers, mlp_ratio, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 