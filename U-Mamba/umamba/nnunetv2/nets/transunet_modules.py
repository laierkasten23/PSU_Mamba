import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout, Linear, Conv2d
from torch.nn.modules.utils import _pair
import numpy as np
from scipy import ndimage


# ---- Utility functions ----
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


# ---- Core Transformer Components ----
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.transformer["num_heads"]
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.transpose_for_scores(self.query(x))
        k = self.transpose_for_scores(self.key(x))
        v = self.transpose_for_scores(self.value(x))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = self.softmax(scores)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(x.size(0), -1, self.all_head_size)
        out = self.out(context)
        return self.proj_dropout(out)


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Mlp(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---- Embeddings ----
class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_embeddings = Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)  # B x hidden x H' x W'
        x = x.flatten(2).transpose(1, 2)  # B x num_patches x hidden
        x = x + self.position_embeddings
        return self.dropout(x)


# ---- Transformer ----
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.embeddings = Embeddings(config, img_size)
        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.transformer["num_layers"])] )
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.norm(x)


# ---- Decoder ----
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3, padding=1)

        in_chs = [head_channels] + config.decoder_channels[:-1]
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, skip_ch)
            for in_ch, out_ch, skip_ch in zip(in_chs, config.decoder_channels, config.skip_channels)
        ])

    def forward(self, hidden):
        B, N, C = hidden.shape
        h = w = int(N ** 0.5)
        x = hidden.transpose(1, 2).contiguous().view(B, C, h, w)
        x = self.conv_more(x)
        for block in self.blocks:
            x = block(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
        )


# ---- Full TransUNet ----
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2):
        super().__init__()
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.transformer(x)
        x = self.decoder(x)
        return self.head(x)


# ---- Configs ----
class DotDict(dict):
    def __getattr__(self, item): return self[item]
    def __setattr__(self, key, value): self[key] = value


def get_b16_config():
    return DotDict({
        "patches": {"size": (16, 16)},
        "hidden_size": 768,
        "transformer": {
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.1,
        },
        "decoder_channels": [256, 128, 64, 16],
        "skip_channels": [512, 256, 64, 16],
        "n_classes": 2,
        "n_skip": 3,
        "classifier": "seg"
    })

CONFIGS = {"ViT-B_16": get_b16_config()}
