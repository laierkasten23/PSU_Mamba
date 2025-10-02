import math
import inspect
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- helpers ----
def _triple(x):
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return tuple(int(v) for v in x)
    return (int(x), int(x), int(x))


# ---- Transformer blocks ----
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x): # [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_dim, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---- 3D patch + positional embeddings ----
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch_size: Tuple[int, int, int]):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x): # [B, C, D, H, W]
        x = self.proj(x) # [B, dim, D', H', W']
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, C], N=D'*H'*W'
        return x, (D, H, W)
    
class PosEmbed3D(nn.Module):
    """Learned pos-embed with trilinear interpolation for arbitrary token grids."""
    def __init__(self, dim: int, ref_grid: Tuple[int, int, int], drop: float = 0.0):
        super().__init__()
        self.ref_grid = tuple(int(v) for v in ref_grid)
        self.pos = nn.Parameter(torch.zeros(1, ref_grid[0] * ref_grid[1] * ref_grid[2], dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)

    @staticmethod
    def _interp(pos: torch.Tensor, old: Tuple[int, int, int], new: Tuple[int, int, int]) -> torch.Tensor:
    # pos: [1, N, C]
        D0, H0, W0 = old
        pos = pos.reshape(1, D0, H0, W0, -1).permute(0, 4, 1, 2, 3) # [1, C, D, H, W]
        pos = F.interpolate(pos, size=new, mode='trilinear', align_corners=False)
        pos = pos.permute(0, 2, 3, 4, 1).reshape(1, new[0]*new[1]*new[2], -1)
        return pos

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]):
        if grid[0] * grid[1] * grid[2] == self.pos.shape[1]:
            pe = self.pos
        else:
            pe = self._interp(self.pos, self.ref_grid, grid)
        return self.drop(x + pe)
    
# ---- 3D decoder ----
class ConvBNReLU3D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = ConvBNReLU3D(in_ch, out_ch)
        self.conv2 = ConvBNReLU3D(out_ch, out_ch)

    def forward(self, x):
        if self.scale_factor != (1, 1, 1):
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup3D(nn.Module):
    def __init__(self, embed_dim: int, decoder_channels=(256, 128, 64, 32), patch_size=(2, 16, 16)):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = ConvBNReLU3D(embed_dim, decoder_channels[0])
        
        # Create blocks with appropriate upsampling factors
        # For patch_size (2, 16, 16), we need total upsampling of (2, 16, 16)
        # We'll do progressive upsampling: (1,2,2), (1,2,2), (1,2,2), (2,2,2)
        upsampling_factors = [
            (1, 2, 2),  # 1x2x2 -> spatial only
            (1, 2, 2),  # 1x2x2 -> spatial only  
            (1, 2, 2),  # 1x2x2 -> spatial only
            (2, 2, 2),  # 2x2x2 -> include depth
        ]
        
        blocks = []
        for i in range(len(decoder_channels) - 1):
            scale_factor = upsampling_factors[i] if i < len(upsampling_factors) else (2, 2, 2)
            blocks.append(DecoderBlock3D(decoder_channels[i], decoder_channels[i + 1], scale_factor))
        self.blocks = nn.ModuleList(blocks)
        self.out_ch = decoder_channels[-1]

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]):
        B, N, C = x.shape
        D, H, W = grid
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.proj(x)
        
        # Apply upsampling blocks
        for blk in self.blocks:
            x = blk(x)
        
        return x

class SegmentationHead3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
    
class TransUNet3DWrapper(nn.Module):
    """Wrapper to handle nnU-Net's inference requirements"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def __call__(self, x):
        output = self.model(x)
        # For inference operations, return the first element (tensor)
        # For training/validation loss, the original list format is used
        if isinstance(output, list) and len(output) == 1:
            return output[0]
        return output
    
    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped model
        if name == 'model':
            return super().__getattr__(name)
        return getattr(self.model, name)


class TransUNet3D(nn.Module):
    """ViT over 3D patches + simple 3D decoder (no CNN-encoder skips)."""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        drop: float = 0.0,
        decoder_channels=(256, 128, 64, 32),
    ):
        super().__init__()
        self.patch_size = _triple(patch_size)
        token_grid = tuple(int(img_size[i] // self.patch_size[i]) for i in range(3))

        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        self.pos_embed = PosEmbed3D(embed_dim, token_grid, drop=drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, drop=drop, attn_drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.decoder = DecoderCup3D(embed_dim, decoder_channels, patch_size)
        self.seg_head = SegmentationHead3D(self.decoder.out_ch, num_classes)

    def forward(self, x): # x: [B, C, D, H, W]
        input_shape = x.shape[2:]  # Store original spatial dimensions
        x, grid = self.patch_embed(x)
        x = self.pos_embed(x, grid)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.decoder(x, grid)
        
        # Ensure output matches input spatial dimensions
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='trilinear', align_corners=False)
        
        logits = self.seg_head(x)
        
        # Check if we're being called from inference operations (torch.flip context)
        # by examining the call stack
        call_stack = inspect.stack()
        in_inference = any('flip' in frame.function or 
                          '_internal_maybe_mirror_and_predict' in frame.function or
                          'predict_sliding_window' in frame.function
                          for frame in call_stack)
        
        # Return tensor for inference operations, list for training/validation loss
        if in_inference:
            return logits
        else:
            return [logits]
    
# Simple test to verify model works in both modes
if __name__ == "__main__":
    # Test model in both training and evaluation modes
    model = TransUNet3D(
        in_channels=1,
        num_classes=2,
        img_size=(160, 128, 112),
        patch_size=(2, 16, 16),
        embed_dim=384,
        depth=2,  # Reduced for testing
        num_heads=6,
        mlp_dim=512,  # Reduced for testing
        decoder_channels=(64, 32, 16, 8)  # Reduced for testing
    )
    
    # Test input
    x = torch.randn(1, 1, 160, 128, 112)
    
    # Test training mode (returns list)
    model.train()
    train_output = model(x)
    print(f"Training mode output type: {type(train_output)}, length: {len(train_output) if isinstance(train_output, list) else 'N/A'}")
    print(f"Training mode output shape: {train_output[0].shape if isinstance(train_output, list) else train_output.shape}")
    
    # Test evaluation mode (returns list)
    model.eval()
    with torch.no_grad():
        eval_output = model(x)
    print(f"Eval mode output type: {type(eval_output)}, length: {len(eval_output) if isinstance(eval_output, list) else 'N/A'}")
    print(f"Eval mode output shape: {eval_output[0].shape if isinstance(eval_output, list) else eval_output.shape}")
    
    # Test wrapper for inference (returns tensor)
    wrapped_model = TransUNet3DWrapper(model)
    with torch.no_grad():
        inference_output = wrapped_model(x)
    print(f"Wrapped inference output type: {type(inference_output)}")
    print(f"Wrapped inference output shape: {inference_output.shape}")
    
    print("Model test passed!")


