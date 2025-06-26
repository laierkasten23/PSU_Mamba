import torch
import torch.nn.functional as F

import os
import numpy as np
import nibabel as nib
import torch
import numpy as np
from sklearn.decomposition import PCA as SKPCA



def get_patch_coords(volume_shape, patch_size, stride=None):
    """
    Generate a list of top-left patch coordinates in a 3D volume.

    Args:
        volume_shape (tuple): Shape of the volume (D, H, W).
        patch_size (tuple): Size of the patch (d, h, w).
        stride (tuple or None): Optional stride. If None, use patch size (no overlap).

    Returns:
        List of (z, y, x) tuples indicating patch start positions.
    """
    if stride is None:
        stride = patch_size

    D, H, W = volume_shape
    d, h, w = patch_size
    sd, sh, sw = stride

    coords = []
    for z in range(0, D - d + 1, sd):
        for y in range(0, H - h + 1, sh):
            for x in range(0, W - w + 1, sw):
                coords.append((z, y, x))
    return coords

def get_patch_coords(patch_size, image_shape):
    """
    Generate patch coordinates based on the patch size and image shape.
    
    Args:
        patch_size (tuple): Size of the patches.
        image_shape (tuple): Shape of the image.
    
    Returns:
        np.ndarray: Coordinates of the patches.
    """
    coords = []
    for z in range(0, image_shape[0] - patch_size[0] + 1, patch_size[0]):
        for y in range(0, image_shape[1] - patch_size[1] + 1, patch_size[1]):
            for x in range(0, image_shape[2] - patch_size[2] + 1, patch_size[2]):
                coords.append((z, y, x))
    return np.array(coords)




def extract_patch(volume, coord, patch_size):
    """
    Extract a patch from a 3D volume.

    Args:
        volume (torch.Tensor): Tensor of shape (C, D, H, W).
        coord (tuple): Starting (z, y, x) coordinate.
        patch_size (tuple): (d, h, w)

    Returns:
        torch.Tensor: Patch of shape (C, d, h, w)
    """
    z, y, x = coord
    d, h, w = patch_size
    return volume[:, z:z+d, y:y+h, x:x+w]


def extract_patches_and_origins(feats, patch_size):
    """
    feats: (B, C, D, H, W)
    Returns: list of (C, d, h, w) patches and their (z, y, x) coordinates
    """
    patches = []
    coords = []
    b, c, D, H, W = feats.shape
    dz, dy, dx = patch_size
    for z in range(0, D - dz + 1, dz):
        for y in range(0, H - dy + 1, dy):
            for x in range(0, W - dx + 1, dx):
                patch = feats[0, :, z:z+dz, y:y+dy, x:x+dx]  # assuming batch size 1 for now
                patches.append(patch)
                coords.append((z, y, x))
    return patches, coords


def compute_local_pca(volume, patch_coords, patch_size, n_components=2, use_sklearn=False):
    """
    Compute local PCA vectors for each patch using either sklearn or torch backend.

    Args:
        volume (torch.Tensor): Shape (C, D, H, W)
        patch_coords (list): List of (z, y, x) tuples
        patch_size (tuple): (d, h, w)
        n_components (int): Number of PCA components
        use_sklearn (bool): If True, use sklearn.PCA; else use torch.linalg.svd

    Returns:
        dict: Mapping from coord to PCA basis (Tensor of shape [n_components, C])
    """
    pca_vectors = {}

    for coord in patch_coords:
        patch = extract_patch(volume, coord, patch_size)  # shape: (C, d, h, w)
        flat = patch.reshape(patch.shape[0], -1).T  # shape: (voxels, channels)

        if use_sklearn:
            try:
                flat_np = flat.cpu().numpy()
                pca = SKPCA(n_components=n_components)
                pca.fit(flat_np)
                vec = torch.tensor(pca.components_, dtype=flat.dtype)
            except Exception as e:
                print(f"[SKLearn PCA] Failed at coord {coord} with error: {e}")
                vec = torch.eye(flat.shape[1])[:n_components]
        else:
            try:
                centered = flat - flat.mean(dim=0, keepdim=True)
                # Use SVD for PCA
                # torch.linalg.svd returns U, S, Vh; we need Vh for PCA components
                # Note: S is singular values, not needed for PCA vectors
                u, s, vh = torch.linalg.svd(centered, full_matrices=False)
                vec = vh[:n_components]
            except Exception as e:
                print(f"[Torch PCA] Failed at coord {coord} with error: {e}")
                vec = torch.eye(flat.shape[1], device=flat.device)[:n_components]

        pca_vectors[coord] = vec

    return pca_vectors


def compute_global_pca(prediction_map, n_components=2):
    """
    Compute global PCA vectors from the prediction map.
    
    Args:
        prediction_map (np.ndarray): The aggregated prediction map.
    Returns:
        np.ndarray: Global PCA vectors.
    """
    from sklearn.decomposition import PCA
    
    # Flatten the prediction map
    flat_map = prediction_map.flatten()
    
    # Perform PCA
    print("DEBUG: Computing global PCA...")
    pca = PCA(n_components=n_components)
    pca.fit(flat_map.reshape(-1, 1))
    
    return pca.components_


def reassemble_patches(patches, coords, volume_shape, patch_size):
    """
    Reassemble patches into a full volume.

    Args:
        patches (list): List of patches.
        coords (list): List of (z, y, x) coordinates for each patch.
        volume_shape (tuple): Shape of the full volume (D, H, W).
        patch_size (tuple): Size of the patches (d, h, w).

    Returns:
        torch.Tensor: Full volume with shape (C, D, H, W).
    """
    C = patches[0].shape[0]
    D, H, W = volume_shape
    d, h, w = patch_size

    full_volume = torch.zeros((C, D, H, W), dtype=patches[0].dtype)

    for patch, coord in zip(patches, coords):
        z, y, x = coord
        full_volume[:, z:z+d, y:y+h, x:x+w] += patch

    return full_volume


