# Based on
# https://github.com/facebookresearch/segment-anything/issues/169#issuecomment-1551120325

import numpy as np


def pad_mask(
    ref_mask: np.ndarray,
    new_height: int,
    new_width: int,
    pad_all_sides: bool = False,
) -> np.ndarray:
    """
    Add padding to an image to make it square.

    Args:
        ref_mask (np.ndarray): The image to be padded.
        new_height (int): The height of the image after resizing.
        new_width (int): The width of the image after resizing.
        pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

    Returns:
        np.ndarray: The padded image.
    """
    pad_height = 256 - new_height
    pad_width = 256 - new_width
    if pad_all_sides:
        padding = (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        )
    else:
        padding = ((0, pad_height), (0, pad_width))

    # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
    return np.pad(ref_mask, padding, mode="constant")
