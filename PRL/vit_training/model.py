import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange

class ViT3DSegmenter(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, img_size=(240, 240, 180), dim=512, depth=6, heads=8, mlp_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        D, H, W = img_size
        self.num_patches = (D // patch_size) * (H // patch_size) * (W // patch_size)
        patch_dim = in_channels * patch_size ** 3

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (c p1 p2 p3)', p1=patch_size, p2=patch_size, p3=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth
        )

        self.head = nn.Sequential(
            nn.Linear(dim, patch_size ** 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x[:, 1:, :]
        x = self.head(x)
        D, H, W = 240 // self.patch_size, 240 // self.patch_size, 180 // self.patch_size
        x = rearrange(x, 'b (d h w) (p1 p2 p3) -> b 1 (d p1) (h p2) (w p3)', d=D, h=H, w=W, p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)
        return x
