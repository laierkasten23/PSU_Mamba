# Based on
# https://github.com/facebookresearch/segment-anything/issues/169#issuecomment-1551120325

import numpy as np
from .pad_mask import pad_mask
from .resize_mask import resize_mask


def reference_to_sam_mask(
    ref_mask: np.ndarray, threshold: int = 127, pad_all_sides: bool = False
) -> np.ndarray:
    """
    Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.

    Args:
        ref_mask (np.ndarray): The grayscale mask to be processed.
        threshold (int, optional): The threshold value for the binarization. Default is 127.
        pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

    Returns:
        np.ndarray: The processed binary mask.
    """

    # Convert a grayscale mask to a binary mask.
    # Values over the threshold are set to 1, values below are set to -1.
    ref_mask = np.clip((ref_mask > threshold) * 2 - 1, -1, 1)

    # Resize to have the longest side 256.
    resized_mask, new_height, new_width = resize_mask(ref_mask)

    # Add padding to make it square.
    square_mask = pad_mask(resized_mask, new_height, new_width, pad_all_sides)

    # Expand SAM mask's dimensions to 1xHxW (1x256x256).
    return np.expand_dims(square_mask, axis=0)
