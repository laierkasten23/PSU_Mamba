import numpy as np
from scipy.spatial import cKDTree


def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + 1e-5) / (union + 1e-5)


def hausdorff_distance(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred_coords = np.argwhere(pred > 0.5)
    target_coords = np.argwhere(target > 0.5)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float('inf')

    pred_tree = cKDTree(pred_coords)
    target_tree = cKDTree(target_coords)

    d1, _ = pred_tree.query(target_coords, k=1)
    d2, _ = target_tree.query(pred_coords, k=1)

    return max(d1.max(), d2.max())

def compute_metrics(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # Dice Coefficient
    dice = dice_coefficient(pred, target)

    # Hausdorff Distance
    hd = hausdorff_distance(pred, target)

    return {
        'dice_coefficient': dice,
        'hausdorff_distance': hd
    }