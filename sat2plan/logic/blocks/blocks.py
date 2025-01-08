import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=4, stride=2, padding=1, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)

        return self.dropout(x) if self.use_dropout else x


class UVCCNNlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, stride=1, padding=1, down=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride,
                      padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, padding_mode="reflect"),
        )

    def forward(self, x):
        return self.conv(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, features, ffn_features, n_heads, rezero=True, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.norm1 = nn.LayerNorm(features, eps=1e-5)
        
        self.atten = nn.MultiheadAttention(
            features, 
            n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(features, eps=1e-5)
        self.ffn = PositionWise(features, ffn_features, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.rezero = rezero
        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.re_alpha = 1

    def forward(self, x):
        residual = x
        
        y = self.norm1(x)
        y, _ = self.atten(y, y, y)
        y = self.dropout1(y)
        x = residual + (y * self.re_alpha if self.rezero else y)

        residual = x
        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout2(y)
        x = residual + (y * self.re_alpha if self.rezero else y)

        return x

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )


class PositionWise(nn.Module):
    def __init__(self, features, ffn_features, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(features, ffn_features, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_features, features, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, n_blocks,
        rezero=True, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(*[
            TransformerBlock(
                features, ffn_features, n_heads, rezero
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result


class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self.projector = nn.Linear(2, features)
        self._height = height
        self._width = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim=2)

        return torch.sin(self.projector(z))


class ViTInput(nn.Module):

    def __init__(
        self, input_features, embed_features, features, height, width,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._height = height
        self._width = width
        #     height, width, embed_features, features)
        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)
        x, y = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))
        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # embed : (1, height * width, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim=2)

        return self.output(result)


class PixelwiseViT(nn.Module):
    def __init__(self, features, n_heads, n_blocks, ffn_features, embed_features, 
                 image_shape, rezero=True, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape

        self.seq_length = image_shape[1] * image_shape[2]
        
        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2]
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                features, ffn_features, n_heads, rezero, dropout
            ) for _ in range(n_blocks)
        ])
        
        self.trans_output = nn.Linear(features, image_shape[0], bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        itokens = x.reshape(batch_size, x.shape[1], -1).transpose(1, 2)
        
        y = self.trans_input(itokens)
        
        for block in self.encoder:
            y = block(y)
            
        otokens = self.trans_output(y)
        return otokens.transpose(1, 2).reshape(batch_size, -1, *self.image_shape[1:])


def calc_tokenized_size(image_shape, token_size):
    if image_shape[1] % token_size[0] != 0:
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        )

    if image_shape[2] % token_size[1] != 0:
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        )

    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])


def img_to_tokens(image_batch, token_size):

    result = image_batch.view(
        (*image_batch.shape[:2], -1, token_size[0], image_batch.shape[3])
    )
    result = result.view((*result.shape[:4], -1, token_size[1]))

    result = result.permute((0, 2, 4, 1, 3, 5))

    return result


def img_from_tokens(tokens):

    result = tokens.permute((0, 3, 1, 4, 2, 5))

    result = result.reshape((*result.shape[:4], -1))

    result = result.reshape((*result.shape[:2], -1, result.shape[4]))

    return result
