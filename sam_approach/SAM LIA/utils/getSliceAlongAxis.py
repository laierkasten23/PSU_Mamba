def getSliceAlongAxis(volume, nSlice, axis):
    """
    Returns a slice from a volume along a given axis

    Parameters:
    volume (ndarray): The input volume
    nSlice (int): The index of the slice to extract
    axis (int): The axis along which to extract the slice (0, 1, or 2)

    Returns:
    ndarray: The extracted slice from the volume

    Raises:
    ValueError: If the axis is not 0, 1, or 2
    """
    if axis == 0:
        return volume[nSlice, :, :]
    elif axis == 1:
        return volume[:, nSlice, :]
    elif axis == 2:
        return volume[:, :, nSlice]
    else:
        raise ValueError("Axis must be 0, 1 or 2")
