import torch
import torch.nn as nn
from nnunetv2.utilities.utils_patch_pca import extract_patches_and_origins, reassemble_patches

class ResidualMambaEncoder(nn.Module):
    def __init__(self, ...):  # Keep the rest of your constructor untouched
        super().__init__()
        # All your setup (same as before)

    def forward(self, x):
        if self.stem is not None:
            conv = self.stem[0]  # BasicResBlock
            mamba = self.stem[1]  # MambaLayer
            rest = self.stem[2:]  # List of remaining blocks

            x = conv(x)

            patch_size = (32, 32, 32)  # You can later pass this via constructor
            patches, coords = extract_patches_and_origins(x, patch_size)
            mamba_outputs = []

            for patch, origin in zip(patches, coords):
                principal_vector = mamba.get_local_pca_vector(origin)
                x_flat, unpermute = mamba.flatten_pca_scan(patch, principal_vector)
                x_norm = mamba.norm(x_flat)
                x_mamba = mamba.mamba(x_norm)
                patch_out = unpermute(x_mamba)
                mamba_outputs.append(patch_out)

            x = reassemble_patches(mamba_outputs, coords, x.shape)

            for block in rest:
                x = block(x)

        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)

        return ret if self.return_skips else ret[-1]
