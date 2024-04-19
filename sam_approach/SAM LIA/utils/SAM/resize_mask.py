# Based on
# https://github.com/facebookresearch/segment-anything/issues/169#issuecomment-1551120325

import numpy as np
import cv2


def resize_mask(
    ref_mask: np.ndarray, longest_side: int = 256
) -> tuple[np.ndarray, int, int]:
    """
    Resize an image to have its longest side equal to the specified value.

    Args:
        ref_mask (np.ndarray): The image to be resized.
        longest_side (int, optional): The length of the longest side after resizing. Default is 256.

    Returns:
        tuple[np.ndarray, int, int]: The resized image and its new height and width.
    """
    height, width = ref_mask.shape[:2]
    if height > width:
        new_height = longest_side
        new_width = int(width * (new_height / height))
    else:
        new_width = longest_side
        new_height = int(height * (new_width / width))

    return (
        cv2.resize(ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST),
        new_height,
        new_width,
    )
