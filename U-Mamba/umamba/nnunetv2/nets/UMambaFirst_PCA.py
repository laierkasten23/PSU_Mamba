import os
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.utilities.pca_utils import extract_patches_and_origins, reassemble_patches



class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, scan_type='x'):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.local_pca_vectors = None
        self.local_pca_coords = None
        self.scan_type = scan_type
        self.register_buffer('global_pca_vector', torch.zeros(3), persistent=True)
        
    def get_local_pca_vector(self, patch_origin):
        idx = self.get_patch_index(patch_origin)
        print("Index of patch origin:", idx)
        if idx is not None:
            return self.local_pca_vectors[idx]
        else:
            return self.global_pca_vector
        
    def set_local_pca_vectors(self, vectors, coords):
        """
        Sets local PCA vectors and their corresponding patch coordinates.
        Accepts:
            vectors: (N, C) for 1 component, or (N, 2, C) for 2 components
            coords:  (N, 3) for (z, y, x) positions
        """
        # Convert to torch if needed
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        vectors = vectors.float()
        coords = coords.long()

        # Validate dimensions
        if vectors.ndim == 2:
            vectors = vectors.unsqueeze(1)  # → (N, 1, C) for consistent handling
        elif vectors.ndim != 3:
            raise ValueError(f"Expected vectors of shape (N, C) or (N, 2, C), got {vectors.shape}")

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords of shape (N, 3), got {coords.shape}")

        self.local_pca_vectors = vectors  # (N, k, C) where k = 1 or 2
        self.local_pca_coords = coords


    def get_patch_index(self, patch_origin):
        print("Patch origin:", patch_origin)
        arr = torch.tensor(patch_origin, dtype=torch.long)
        if self.local_pca_coords is None:
            print("[MambaLayer] No local PCA coordinates set. Returning None.")
            return None
        # Ensure coords are also long/int
        coords = self.local_pca_coords
        if coords.dtype != torch.long:
            coords = coords.long()
        matches = torch.all(coords == arr, dim=1)
        idx = torch.where(matches)[0]
        return idx[0].item() if len(idx) > 0 else None

    def get_global_pca_vector(self):
        """
        Get the global PCA vector used in the Mamba layer.
        Returns:
            torch.Tensor: (3,) vector representing the global PCA direction.
        """
        if self.global_pca_vector is None or self.global_pca_vector.numel() == 0:
            raise ValueError("Global PCA vector is not set. Please set it using set_global_pca_vectors()")
        return self.global_pca_vector


    def set_global_pca_vector(self, vector):
        """
        Set the global PCA vector used in the Mamba layer.
        Accepts a single vector of shape (3,).
        """
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)

    def set_global_pca_vectors(self, vector):
        """
        Set one or more scan vectors used in the Mamba layer.

        Accepts a single vector of shape (3,) or multiple vectors of shape (N, 3).

        Args:
            vector (np.ndarray or torch.Tensor): (3,) or (N, 3)
        """
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)

        vector = vector.to(self.global_pca_vector.device, dtype=self.global_pca_vector.dtype)

        if vector.ndim == 1 and vector.shape[0] == 3:
            # Single vector
            self.global_pca_vector.copy_(vector)
        elif vector.ndim == 2 and vector.shape[1] == 3:
            # Multiple vectors: use the first as representative (or update your logic to handle all)
            self.global_pca_vector.copy_(vector[0])  # <- Replace this line if you want multi-vector support
        else:
            raise ValueError(f"Expected shape (3,) or (N, 3), got {tuple(vector.shape)}")

    def set_pca_patch_size(self, patch_size):
        """
        Set the patch size for local PCA scans.
        Args:   
            patch_size (tuple): (d, h, w) dimensions of the patch.
        """
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != 3:
                raise ValueError(f"Expected patch_size to be a tuple of length 3, got {len(patch_size)}")
            self.pca_patch_size = tuple(patch_size)
        elif isinstance(patch_size, int):
            self.pca_patch_size = (patch_size, patch_size, patch_size)
        else:
            raise ValueError(f"Expected patch_size to be a tuple or int, got {type(patch_size)}")
     
    def pad_to_fit(volume, patch_size):
        _, _, D, H, W = volume.shape
        pad_D = (patch_size[0] - D % patch_size[0]) % patch_size[0]
        pad_H = (patch_size[1] - H % patch_size[1]) % patch_size[1]
        pad_W = (patch_size[2] - W % patch_size[2]) % patch_size[2]
        pad = (0, pad_W, 0, pad_H, 0, pad_D)  # last dimension first
        return F.pad(volume, pad), pad   

    def flatten_for_scan(self, x, scan_type='x', patch_origin=None):
        print(f"[MambaLayer] Flattening for scan type: {scan_type}, patch origin: {patch_origin}")
        B, C, D, H, W = x.shape

        # can delete this later as now we handle 3 different functions
        if scan_type in ['global_pca', 'local_pca', 'pca']:
            print("WARNING: SHOULD NEVER ENTER HERE")
            exit() #! delete this block, should never enter here in theory. 
            vector = self.get_local_pca_vector(patch_origin) if patch_origin else self.global_pca_vector
            return self.flatten_pca_scan(x, vector)

        # original: (B, C, D, H, W) = (B, C, Z, Y, X) where X = D, Y = H, Z = W
        permute, unpermute = {
            'x':  ((0, 1, 4, 3, 2), lambda t: t.reshape(B, C, W, H, D).permute(0, 1, 4, 3, 2)), # x -> y -> z
            'y':  ((0, 1, 3, 4, 2), lambda t: t.reshape(B, C, H, W, D).permute(0, 1, 4, 2, 3)), # y -> x -> z
            #'z':  ((0, 1, 3, 4, 2), lambda t: t.permute(0, 1, 3, 4, 2).reshape(B, C, -1)),  # z fastest, then x, then y               # z -> y -> x
            'z':  ((0, 1, 2, 4, 3), lambda t: t.reshape(B, C, D, W, H).permute(0, 1, 2, 4, 3)),  # z fastest, then x, then y
            'diag': (None, lambda t: t.transpose(-1, -2).reshape(B, C, D, H, W)),  # fallback
        }.get(scan_type, (None, lambda t: t.transpose(-1, -2).reshape(B, C, D, H, W)))

        if scan_type == 'yz-diag':
            print("Using yz-diag-scan")
            x_perm = x.permute(0, 1, 4, 3, 2)  # (B, C, X, Y, Z)
            X, Y, Z = x_perm.shape[2:]
            z_coords, y_coords = torch.meshgrid(
                torch.arange(Z, device=x.device),
                torch.arange(Y, device=x.device),
                indexing='ij'
            )
            
            diag_order = torch.argsort((z_coords + y_coords).flatten()) #! changed
            x_flat = x_perm.reshape(B, C, X, Y * Z)[:, :, :, diag_order]
            flatten = x_flat.reshape(B, C, -1).transpose(-1, -2)
            unpermute = lambda t: t.reshape(B, C, X, Y, Z).permute(0, 1, 4, 3, 2)
            return flatten, unpermute

        elif scan_type == 'xy-diag':
            x_perm = x.permute(0, 1, 2, 4, 3)  # (B, C, Z, X, Y)
            D, X, Y = x_perm.shape[2:]
            x_coords, y_coords = torch.meshgrid(
                torch.arange(X, device=x.device),
                torch.arange(Y, device=x.device),
                indexing='ij'
            )
            
            diag_order = torch.argsort((x_coords + y_coords).flatten()) #! changed
            x_flat = x_perm.reshape(B, C, D, X * Y)[:, :, :, diag_order]
            flatten = x_flat.reshape(B, C, -1).transpose(-1, -2)
            unpermute = lambda t: t.reshape(B, C, D, X, Y).permute(0, 1, 2, 4, 3)
            return flatten, unpermute

        # Default permute-based scan
        x_perm = x.permute(*permute) if permute else x
        x_flat = x_perm.reshape(B, C, -1).transpose(-1, -2)
        
        return x_flat, unpermute


    def flatten_pca_scan(self, x, global_pca_vector):
        """
        Flatten tensor using a global PCA direction.

        Args:
            x: Tensor of shape (B, C, D, H, W)
            global_pca_vector: Tensor or ndarray of shape (3,) or (1, 3)

        Returns:
            x_flat: (B, N, C)
            unpermute: function to reverse flattening
        """
        print("[MambaLayer] Flattening using PCA scan path...")

        if global_pca_vector is None:
            print("[MambaLayer] Principal vector is None. Skipping PCA flatten.")
            return None, None

        # Normalize and ensure shape is (3,)
        if isinstance(global_pca_vector, np.ndarray):
            global_pca_vector = torch.from_numpy(global_pca_vector)
        if global_pca_vector.ndim == 2:
            global_pca_vector = global_pca_vector[0]
        global_pca_vector = global_pca_vector.to(x.device, dtype=torch.float32)

        print("[MambaLayer] Using PCA direction:", global_pca_vector.cpu().numpy())

        B, C, D, H, W = x.shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=x.device),
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3).float()  # (N, 3)

        # Project coords onto vector
        projections = torch.matmul(coords, global_pca_vector)  # (N,)
        
        sorted_idx = torch.argsort(projections)

        # Flatten x
        x_flat_list = [x[b].reshape(C, -1)[:, sorted_idx].permute(1, 0) for b in range(B)]  # (N, C)
        x_flat = torch.stack(x_flat_list)  # (B, N, C)

        def unpermute(t_flat):
            if t_flat.ndim != 5:
                raise ValueError(f"[unpermute] Expected shape (B, C, D, H, W), got {t_flat.shape}")

            t_reshaped = t_flat.reshape(B, C, -1)  # (B, C, N)
            out = torch.zeros_like(t_reshaped)
            for b in range(B):
                out[b, :, sorted_idx] = t_reshaped[b]
            return out.reshape(B, C, D, H, W)


        return x_flat, unpermute


    
    def flatten_local_pca_scan2(self, x):
        """
        x: (B, C, D, H, W)
        Returns:
            x_flat: (B, N_total, C)
            unpermute: to map flattened outputs back to (B, C, D, H, W)
        """
        
        if self.local_pca_vectors is None and self.local_pca_coords is None:
            # In first epoch fall bak to global x scan as vectors are not set yet
            print("[MambaLayer] Using fallback flattening for first epoch.")
            print("SHape x", x.shape)
            return self.flatten_for_scan(x, scan_type='x')

        print("[MambaLayer] Flattening using local PCA scan path...")
        B, C, D, H, W = x.shape
        
        full_shape = (240, 240, 180)
        crop_origin = [ (full - crop) // 2 for full, crop in zip(full_shape, (D, H, W)) ]
        z0, y0, x0 = crop_origin
        
        device = x.device
        patch_size = self.pca_patch_size
        coords = self.local_pca_coords  # (N, 1, 3)
        vectors = self.local_pca_vectors  # (N, 3)
        #print("Shaape of vectors ", self.local_pca_vectors.shape)
        #print("Shape of coords ", self.local_pca_coords.shape)

        expected_num_patches = len(coords)
        voxels_per_patch = patch_size[0] * patch_size[1] * patch_size[2]
        total_voxels = expected_num_patches * voxels_per_patch
        
        x_flat = torch.zeros((B, total_voxels, C), device=device)

        x_flat_batches = []
        index_batches = []

        # Process each sample in the batch
        for b in range(B):
            offset = 0
            flat_list = []
            index_list = []
            
            # Extract patches and their coordinates
            for i in range(len(coords)):
                z0, y0, x0 = coords[i].tolist()
                z1, y1, x1 = z0 + patch_size[0], y0 + patch_size[1], x0 + patch_size[2]

                patch = x[b, :, z0:z1, y0:y1, x0:x1]  # (C, dz, dy, dx) = (32, 16, 16, 16)
                print("Patch shape:", patch.shape, "at coords:", coords[i])

                # Do not include patches that do not match the expected size (C, dz, dy, dx)
                if patch.shape[1:] != torch.Size(patch_size):  # safety
                    continue

                zz, yy, xx = torch.meshgrid(
                    torch.arange(patch_size[0], device=device),
                    torch.arange(patch_size[1], device=device),
                    torch.arange(patch_size[2], device=device),
                    indexing='ij'
                )
                # Tensor representing the spatial position of each voxel in the patch -> shift subpatch to correct global position 
                coords_patch = torch.stack([zz + z0, yy + y0, xx + x0], dim=-1).reshape(-1, 3).float() # ([4096, 3]) (N,3) N: total number of voxels in the patch
                #print("Patch coordinates shape:", coords_patch.shape)
                # project each voxel coordinate in a patch onto a principal direction vector
                # Each value in proj represents the position of a voxel along the principal direction
                proj = torch.matmul(coords_patch, vectors[i].squeeze(0).to(device)) # (N)
                sort_idx = torch.argsort(proj) # (N)
                #print("Patch projection shape:", proj.shape, "Sort index shape:", sort_idx.shape)
                
                #print("Patch shape before flattening:", patch.reshape(C, -1).shape)
                
                # patch.reshape(C, -1).shape = torch.Size([C, N])
                patch_flat = patch.reshape(C, -1)[:, sort_idx].T  # (voxels, C)
                #print("Patch flat shape:", patch_flat.shape)
                
                flat_list.append(patch_flat)

                index_list.append((sort_idx, coords_patch.long()))

            
            if flat_list:
                x_flat_batches.append(torch.cat(flat_list, dim=0))  # (N_total, C)
                print("x_flat_batches shape:", x_flat_batches[-1].shape)
  
            else:
                x_flat_batches.append(torch.zeros((1, C), device=device))
            
        x_flat = torch.stack(x_flat_batches)  # (B, N_total, C)
        print("Final x_flat shape:", x_flat.shape)
        

        def unpermute(t):
            out = torch.zeros((B, C, D, H, W), device=t.device)
            for b in range(B):
                offset = 0
                for i in range(len(coords)):
                    z0, y0, x0 = coords[i].tolist()
                    dz, dy, dx = patch_size
                    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

                    num_vox = dz * dy * dx
                    sub = t[b, offset:offset + num_vox].T.reshape(C, dz, dy, dx)
                    out[b, :, z0:z1, y0:y1, x0:x1] = sub
                    offset += num_vox
            return out

        return x_flat, unpermute




    def flatten_local_pca_scan(self, x):
        """
        Flatten using local PCA vectors applied to sub-patches.

        Args:
            x (Tensor): (B, C, D, H, W)

        Returns:
            x_flat (Tensor): (B, N_total, C)
            unpermute (function): to map flattened outputs back to (B, C, D, H, W)
        """
        print("[MambaLayer] Flattening using local PCA scan path...")
        B, C, D, H, W = x.shape
        device = x.device
        patch_size = self.pca_patch_size
        dz, dy, dx = patch_size

        # Handle fallback case for epoch 0
        if self.local_pca_vectors is None or self.local_pca_coords is None:
            print("[MambaLayer] Local PCA vectors/coords not available. Falling back to x scan.")
            return self.flatten_for_scan(x, scan_type='x')

        # === Step 1: Compute crop origin from full shape (assuming center crop)
        full_shape = (240, 240, 180)
        crop_origin = [(f - p) // 2 for f, p in zip(full_shape, (D, H, W))]
        z0, y0, x0 = crop_origin
        
        # === Step 2: Select valid coords that fit inside this crop
        coords = self.local_pca_coords
        vectors = self.local_pca_vectors

        within_crop = (
            (coords[:, 0] >= z0) & (coords[:, 0] + dz <= z0 + D) &
            (coords[:, 1] >= y0) & (coords[:, 1] + dy <= y0 + H) &
            (coords[:, 2] >= x0) & (coords[:, 2] + dx <= x0 + W)
        )

        coords = coords.to(device)
        coords_crop = coords[within_crop] - torch.tensor([z0, y0, x0], device=device)
        vectors_crop = vectors[within_crop]
        
        # Pre-allocate flat tensor and voxel counter
        x_flat = torch.zeros((B, D * H * W, C), device=device)
        filled_mask = torch.zeros((B, D, H, W), dtype=torch.bool, device=device) # to track filled voxels


        x_flat_batches = []

        for b in range(B):

            for i in range(coords_crop.shape[0]):
                z, y, x_ = coords_crop[i].tolist()
                patch = x[b, :, z:z+dz, y:y+dy, x_:x_+dx]  # (C, dz, dy, dx)

                if patch.shape[1:] != torch.Size(patch_size):
                    continue

                zz, yy, xx = torch.meshgrid(
                    torch.arange(dz, device=device),
                    torch.arange(dy, device=device),
                    torch.arange(dx, device=device),
                    indexing='ij'
                )

                coords_patch = torch.stack([zz + z, yy + y, xx + x_], dim=-1).reshape(-1, 3).float()
                #print("Patch coordinates shape:", coords_patch.shape)
                
                # Project and sort
                proj = torch.matmul(coords_patch, vectors_crop[i].squeeze(0).to(device))  # (N,)
                sort_idx = torch.argsort(proj) # (N)
                #print("Patch projection shape:", proj.shape, "Sort index shape:", sort_idx.shape)  
                #print("Patch shape before flattening:", patch.reshape(C, -1).shape)
                

                patch_flat = patch.reshape(C, -1)[:, sort_idx].T  # (N_voxels, C)
                #print("Patch flat shape:", patch_flat.shape)
                
                # Flattened voxel indices in global space
                # Convert 3D patch coordinates to linear indices
                global_coords = coords_patch.long() # (N_voxels, 3)
                # map 3D coordinates to a single index in a flattened array
                lin_idx = global_coords[:, 0] * (H * W) + global_coords[:, 1] * W + global_coords[:, 2]  # (N,)

                # Fill into preallocated tensor
                x_flat[b, lin_idx] = patch_flat
                filled_mask[b, global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]] = True

        #print(f"[MambaLayer] Final x_flat shape: {x_flat.shape}")
                            



        def unpermute(t):
            out = torch.zeros((B, C, D, H, W), device=t.device, dtype=t.dtype)
            for b in range(B):
                filled_voxels = filled_mask[b].nonzero(as_tuple=False)  # (N, 3)
                lin_idx = filled_voxels[:, 0] * (H * W) + filled_voxels[:, 1] * W + filled_voxels[:, 2]
                out[b, :, filled_voxels[:, 0], filled_voxels[:, 1], filled_voxels[:, 2]] = t[b, lin_idx].T
            return out



        print(f"[MambaLayer] Final x_flat shape: {x_flat.shape}")
        return x_flat, unpermute
    
    
    
    @staticmethod
    def flatten_local_pca_scan_tracked(x, coords_list, vectors_list, patch_size, full_shape=(240,240,180)):
        """
        Flatten 3D input using local PCA projections with tracked voxel positions.
        
        Args:
            x           : (B, C, D_crop, H_crop, W_crop) input crop
            coords_list : (N, 3) starting patch coords in full volume space
            vectors_list: (N, 3) PCA vector per patch
            patch_size  : (dz, dy, dx)
            full_shape  : full input shape before cropping (default 240x240x180)

        Returns:
            x_flat: (B, N_total, C)
            unpermute: function to reconstruct (B, C, D, H, W)
        """
        B, C, D_crop, H_crop, W_crop = x.shape
        dz, dy, dx = patch_size

        crop_origin = [(f - c) // 2 for f, c in zip(full_shape, (D_crop, H_crop, W_crop))]
        z0c, y0c, x0c = crop_origin

        flat_all = []
        index_map_all = []

        for b in range(B):
            flat_patches = []
            voxel_indices = []

            for coord, vec in zip(coords_list, vectors_list):
                # shift patch into cropped space
                zc, yc, xc = coord[0] - z0c, coord[1] - y0c, coord[2] - x0c

                # skip if patch doesn't fit inside crop
                if not (0 <= zc < D_crop - dz + 1 and
                        0 <= yc < H_crop - dy + 1 and
                        0 <= xc < W_crop - dx + 1):
                    continue

                patch = x[b, :, zc:zc+dz, yc:yc+dy, xc:xc+dx]
                zz, yy, xx = torch.meshgrid(
                    torch.arange(dz, device=x.device),
                    torch.arange(dy, device=x.device),
                    torch.arange(dx, device=x.device),
                    indexing='ij'
                )
                coords_patch = torch.stack([zz + zc, yy + yc, xx + xc], dim=-1).reshape(-1, 3).float()
                proj = torch.matmul(coords_patch, vec.to(x.device))
                sort_idx = torch.argsort(proj)

                patch_flat = patch.reshape(C, -1)[:, sort_idx].T
                flat_patches.append(patch_flat)

                coords_sorted = coords_patch[sort_idx.long()].long()
                lin_idx = coords_sorted[:, 0] * H_crop * W_crop + coords_sorted[:, 1] * W_crop + coords_sorted[:, 2]
                voxel_indices.append(lin_idx)

            if flat_patches:
                flat_cat = torch.cat(flat_patches, dim=0)
                idx_cat = torch.cat(voxel_indices, dim=0)
            else:
                flat_cat = torch.zeros((1, C), device=x.device)
                idx_cat = torch.zeros((1,), dtype=torch.long, device=x.device)

            flat_all.append(flat_cat)
            index_map_all.append(idx_cat)

        x_flat = torch.stack(flat_all)       # (B, N_total, C)
        index_map = torch.stack(index_map_all)  # (B, N_total)

        def unpermute(t_flat):
            out = torch.zeros((B, C, D_crop * H_crop * W_crop), device=t_flat.device, dtype=t_flat.dtype)
            for b in range(B):
                out[b, :, index_map[b]] = t_flat[b].T
            return out.reshape(B, C, D_crop, H_crop, W_crop)

        return x_flat, unpermute






    def forward(self, x):
        
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        B, C, D, H, W = x.shape
        assert C == self.dim

        if self.scan_type == 'local_pca':
            #x_flat, unpermute = self.flatten_local_pca_scan(x)
            x_flat, unpermute = self.flatten_local_pca_scan_tracked(x, self.local_pca_coords, self.local_pca_vectors, self.pca_patch_size, full_shape=(240, 240, 180))

            use_custom_unpermute = True
        elif self.scan_type == 'global_pca':
            x_flat, unpermute = self.flatten_pca_scan(x, self.global_pca_vector)
            use_custom_unpermute = False
        elif self.scan_type in ['x', 'y', 'z', 'diag', 'yz-diag', 'xy-diag']:
            x_flat, unpermute = self.flatten_for_scan(x, self.scan_type)
            use_custom_unpermute = False
        else:
            raise ValueError(f"Unknown scan type: {self.scan_type}")

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        
        if use_custom_unpermute:
            return unpermute(x_mamba)
        else:
            out = x_mamba.transpose(-1, -2).reshape(B, C, *x.shape[2:]) #! CANNOT change this, otherwise x scan is wrong!!!
            #out_flat = out.permute(0, 2, 3, 4, 1).reshape(B, D*H*W, C) #! flat for global scan
            return unpermute(out) #  why does "x" work with flattened and unflattened version???



class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 scan_type: str = 'x'  # 'x', 'y', 'z', 'yz-diag', 'xy-diag', 'pca', 'global_pca #  
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
                                         
        

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        #feature_map_size = input_size[::-1] # ! CHANGE 
        feature_map_size = input_size 
        #print("feature_map_size Input size", feature_map_size)
        
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
        #print("do_channel_token", do_channel_token)
        #print("feature_map_sizeS size", feature_map_sizes)
        

        #print(f"feature_map_sizes: {feature_map_sizes}")
        #print(f"do_channel_token: {do_channel_token}")
        
        self.scan_type = scan_type
        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            MambaLayer(
                dim=stem_channels,
                scan_type=self.scan_type
                # Not channel_token: scan path will be used
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        #self.dropout_op = dropout_op
        #self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        
        
    def extract_stem_features(self, x):
        # Apply only the stem before any downsampling
        if self.stem is not None:
            conv = self.stem[0]
            x = conv(x)
        return x  # shape: (B, C, D, H, W)
        
    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]



    def compute_conv_feature_map_size(self, input_size):
        #print("Input size encoder initial", input_size)
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
            #print("Input size encoder", input_size)

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        stages = []

        upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            #print("encoder.output_channels[-s]", encoder.output_channels[-s])
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None,
                 scan_type: str = 'x', #, 'global_pca', 'local_pca', 'pca', 'x', 'y', 'z', 'diag' (yz-diag, xy-diag, etc. are not implemented yet!
                 pca_patch_size=None,
                 local_pca_vectors=None,
                 local_pca_coords=None,
                 global_pca_vector=None,
                 output_folder=None,
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1    

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1


        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels, 
            scan_type=scan_type
        )
        self.scan_type = scan_type
        self.pca_patch_size = pca_patch_size
        self.local_pca_vectors = local_pca_vectors
        self.local_pca_coords = local_pca_coords
        self.global_pca_vector = global_pca_vector
        self.output_folder = output_folder
        
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def extract_stem_features(self, x):
        return self.encoder.extract_stem_features(x)
    
    def set_global_pca_vectors(self, pca_vectors: torch.Tensor):
        """
        Sets PCA scan vectors (1 or more) to all MambaLayer modules.

        Args:
            pca_vectors (torch.Tensor): Shape (n_components, 3)
        """
        if pca_vectors.ndim != 2 or pca_vectors.shape[1] != 3:
            raise ValueError(f"Expected shape (n_components, 3), got {pca_vectors.shape}")

        # Normalize all vectors
        normalized_vectors = torch.nn.functional.normalize(pca_vectors, dim=1)

        for module in self.modules():
            if isinstance(module, MambaLayer):
                module.set_global_pca_vectors(normalized_vectors)

                
    def set_local_pca_vectors(self, vectors, coords):
        """
        Broadcast the local PCA vectors and patch coords to all MambaLayers in the model.
        """
        for module in self.modules():
            if isinstance(module, MambaLayer):
                module.set_local_pca_vectors(vectors, coords)
                
    def set_pca_patch_size(self, patch_size):
        """
        Set the PCA patch size for all MambaLayer modules.
        
        Args:
            patch_size (tuple): (d, h, w) dimensions of the patch.
        """
        for module in self.modules():
            if isinstance(module, MambaLayer):
                module.set_pca_patch_size(patch_size)


    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_first_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True, 
        scan_type: str = 'global',
        pca_patch_size=None,
        local_pca_vectors=None,
        local_pca_coords=None,
        global_pca_vector=None,
        output_folder=None
    ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc   # create a network class based on the segmentation_network_class_name
    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # Initialize the model
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        scan_type=scan_type,
        pca_patch_size=pca_patch_size,
        local_pca_vectors=local_pca_vectors,
        local_pca_coords=local_pca_coords,
        global_pca_vector=global_pca_vector,
        output_folder=output_folder, 
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    
    # Set local PCA vectors and coords here
    if local_pca_vectors is not None and local_pca_coords is not None:
        for module in model.modules():
            if isinstance(module, MambaLayer):
                module.set_local_pca_vectors(local_pca_vectors, local_pca_coords)
    
    # set scan type
    for module in model.modules():
        if isinstance(module, MambaLayer):
            module.scan_type = scan_type
            if scan_type == 'global_pca' and global_pca_vector is not None:
                module.set_global_pca_vectors(global_pca_vector)
    
    # Set PCA patch size if provided
    if pca_patch_size is not None:
        for module in model.modules():
            if isinstance(module, MambaLayer):
                module.pca_patch_size = pca_patch_size

    # Save the model configuration to output folder if specified
    if output_folder is not None:
        print(f"Saving model configuration to {output_folder}")
    
    # print scan type and PCA vectors
    print(f"Scan type set to: {scan_type}")
        
    


    return model
