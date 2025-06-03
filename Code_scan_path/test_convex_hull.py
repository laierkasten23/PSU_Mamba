import numpy as np
import torch
from nnunetv2.training.data_augmentation.custom_transforms.convex_hull_transform import ConvexHullTransform

def test_create_convex_hull():
    # Create a dummy segmentation
    seg = torch.zeros((1, 10, 10, 10))
    seg[0, 3:7, 3:7, 3:7] = 1  # Create a cube in the center

    transform = ConvexHullTransform(margin=2)
    hull_volume, hull, points = transform.create_convex_hull(seg)

    # Check that the hull volume is not empty
    assert np.sum(hull_volume) > 0, "Hull volume should not be empty"

    # Check that the points are within the expected range
    assert points.shape[0] > 0, "Points should not be empty"

def test_create_volume_within_convex_hull():
    # Create a dummy segmentation
    seg = torch.zeros((1, 10, 10, 10))
    seg[0, 3:7, 3:7, 3:7] = 1  # Create a cube in the center

    transform = ConvexHullTransform(margin=2)
    volume_within_hull = transform.create_volume_within_convex_hull(seg)

    # Check that the volume within the hull is not empty
    assert np.sum(volume_within_hull) > 0, "Volume within hull should not be empty"

    # Check that the volume within the hull has the expected shape
    assert volume_within_hull.shape == seg.shape[1:], "Volume within hull should have the same shape as the input segmentation"

def test_convex_hull_transform():
    # Create a dummy data and segmentation
    data = np.random.rand(1, 1, 10, 10, 10)
    seg = torch.zeros((1, 10, 10, 10))
    seg[0, 3:7, 3:7, 3:7] = 1  # Create a cube in the center

    transform = ConvexHullTransform(margin=2)
    data_dict = {'data': data, 'seg': seg}

    # Apply the transform
    transformed_data_dict = transform(**data_dict)

    # Check that the transformed data is not empty
    assert np.sum(transformed_data_dict['data']) > 0, "Transformed data should not be empty"

    # Check that the transformed data has the expected shape
    assert transformed_data_dict['data'].shape == data.shape, "Transformed data should have the same shape as the input data"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])