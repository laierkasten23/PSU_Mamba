import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import argparse

'''
Sample usage
python3 preprocess.py -i interpol_2d_xy -shape '(257, 384)'   or 
python3 preprocess.py -i interpol_3d -shape '(257, 384, 384)'   
'''

def interpolate_2d(image, target_shape):
    """
    Interpolate a 2D image to the target shape using bilinear interpolation.
    
    Parameters:
        image (ndarray): 2D input image.
        target_shape (tuple): Target shape (height, width).
    
    Returns:
        ndarray: Interpolated image using spline interpolation of order 1.
    """
    return ndimage.zoom(image, (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]), order=1)

def interpolate_2d_entire_vol(image, target_shape, direction = 'z'):
    """
    Interpolate a 3D image using 2D slices along direction to the target shape.
    
    Parameters:
        image (ndarray): 3D input image.
        target_shape (tuple): Target shape (height, width).
    
    Returns:
        ndarray: Interpolated image using spline interpolation of order 1.
    """
    if direction == 'z':
        new_volume = np.zeros((target_shape[0], target_shape[1], image.shape[2]))
        for i in range(image.shape[2]):
            new_volume[:,:,i] = interpolate_2d(image[:,:,i], target_shape)
    elif direction == 'y':
        new_volume = np.zeros((target_shape[0], image.shape[1], target_shape[1]))
        for i in range(image.shape[1]):
            new_volume[:,i,:] = interpolate_2d(image[:,i,:], target_shape)
    elif direction == 'x':
        new_volume = np.zeros((image.shape[0], target_shape[0], target_shape[1]))
        for i in range(image.shape[0]):
            new_volume[i,:,:] = interpolate_2d(image[i,:,:], target_shape)
    else:
        raise ValueError("Invalid direction. Use 'x', 'y' or 'z'.")
    
    return new_volume

def interpolate_3d(image, target_shape):
    """
    Interpolate a 3D image to the target shape using trilinear spline interpolation of order 1.

    Parameters:
        image (ndarray): 3D input image.
        target_shape (tuple): Target shape (depth, height, width).

    Returns:
        ndarray: Interpolated image.
    """
    return ndimage.zoom(image, (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1], target_shape[2] / image.shape[2]), order=1)


def zero_pad_2d(image, target_shape):
    """
    Zero pad a 2D image to the target shape.
    
    Parameters:
        image (ndarray): 2D input image.
        target_shape (tuple): Target shape (height, width).
    
    Returns:
        ndarray: Zero-padded image.
    """
    pad_height = target_shape[0] - image.shape[0]
    pad_width = target_shape[1] - image.shape[1]
    return np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

def zero_pad_2d_entire_vol(image, target_shape, direction = 'z'):
    """
    Zero pad a 3D image using 2D slices along direction to the target shape.
    
    Parameters:
        image (ndarray): 3D input image.
        target_shape (tuple): Target shape (height, width).
    
    Returns:
        ndarray: Zero-padded image.
    """
    if direction == 'z':
        new_volume = np.zeros((target_shape[0], target_shape[1], image.shape[2]))
        for i in range(image.shape[2]):
            new_volume[:,:,i] = zero_pad_2d(image[:,:,i], target_shape)
    elif direction == 'y':
        new_volume = np.zeros((target_shape[0], image.shape[1], target_shape[1]))
        for i in range(image.shape[1]):
            new_volume[:,i,:] = zero_pad_2d(image[:,i,:], target_shape)
    elif direction == 'x':
        new_volume = np.zeros((image.shape[0], target_shape[0], target_shape[1]))
        for i in range(image.shape[0]):
            new_volume[i,:,:] = zero_pad_2d(image[i,:,:], target_shape)
    else:
        raise ValueError("Invalid direction. Use 'x', 'y' or 'z'.")
    
    return new_volume

def zero_pad_3d(image, target_shape):
    """
    Zero pad a 3D image to the target shape.

    Parameters:
        image (ndarray): 3D input image.
        target_shape (tuple): Target shape (depth, height, width).

    Returns:
        ndarray: Zero-padded image.
    """
    pad_depth = target_shape[0] - image.shape[0]
    pad_height = target_shape[1] - image.shape[1]
    pad_width = target_shape[2] - image.shape[2]
    return np.pad(image, ((0, pad_depth), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)


def center_and_resize_2d(image, output_shape, direction='z'):
    """
    Center and resize a 3D image to the desired 2D output shape (slicewise)

    Parameters:
    - image: 3D NumPy array representing the original image.
    - output_shape: Tuple specifying the desired 2D output shape.
    - direction: 'z' or 'y' direction for resizing.

    Returns:
    - centered_resized_image: Centered and resized 3D image.
    """

    if direction not in ['x', 'y', 'z']:
        raise ValueError("Invalid direction. Choose 'x', 'y' or 'z'.")

    # Get the shape of the original image
    original_shape = image.shape

    # Calculate the center of the original image
    center = [dim // 2 for dim in original_shape]

    # Calculate the new start and end indices after resizing for each dimension
    start_indices = [max(0, center[i] - output_shape[i] // 2) for i in range(2)]
    end_indices = [min(original_shape[i], center[i] + output_shape[i] // 2) for i in range(2)]

    # Initialize the slices for each dimension
    slices = [slice(start, end) for start, end in zip(start_indices, end_indices)]

    if direction == 'x':
        slices.insert(0, slices)
        slices[0] = np.arange(original_shape[0]) 
    elif direction == 'y':
        slices.insert(1, slices)
        slices[1] = np.arange(original_shape[1])
    else:
        slices.insert(2, slices)
        slices[2] = np.arange(original_shape[2])
    
    # Slice the original image to get the centered and resized image
    centered_resized_image = image[slices[0], slices[1], slices[2]]

    return centered_resized_image


def center_and_resize_3d(image, output_shape):
    """
    Center and resize a 3D image to the desired 3D output shape.

    Parameters:
    - image: 3D NumPy array representing the original image.
    - output_shape: Tuple specifying the desired 3D output shape.

    Returns:
    - centered_resized_image: Centered and resized 3D image.
    """

    # Get the shape of the original image
    original_shape = image.shape

    # Calculate the center of the original image
    center = [dim // 2 for dim in original_shape]

    # Calculate the new start and end indices after resizing
    start_indices = [max(0, center[i] - output_shape[i] // 2) for i in range(3)]
    end_indices = [min(original_shape[i], center[i] + output_shape[i] // 2) for i in range(3)]
    
    # Slice the original image to get the centered and resized image
    centered_resized_image = image[start_indices[0]:end_indices[0],
                                  start_indices[1]:end_indices[1],
                                  start_indices[2]:end_indices[2]]

    return centered_resized_image


def process_subject(data_path, subject_name, method, target_shape, save_data = True):
    
    subject_folder = os.path.join(data_path, subject_name)
    print("subject folder: " + subject_folder)

    # Load T1.nii data and mask.nii data
    t1_file = os.path.join(subject_folder, 'T1.nii')
    mask_file = os.path.join(subject_folder, 'mask.nii')

    t1_data = nib.load(t1_file).get_fdata()
    mask_data = nib.load(mask_file).get_fdata()

    # preprocess mask data to just have 0 and 1 as labels
    mask_data[mask_data > 0] = 1

    # Define target shape
    if len(target_shape) == 2:
        if method == 'interpol_3d' or method == 'zero_pad_3d' or method == 'center_and_resize_3d':
            raise ValueError("Invalid target shape. Use a 3D target shape for 3D methods.")
        
    else: 
        if method == 'interpol_2d_xy' or method == 'zero_pad_2d_xy' or method == 'center_and_resize_2d_xy' :
            target_shape = (target_shape[0], target_shape[1])
        elif method == 'interpol_2d_xz' or method == 'zero_pad_2d_xz' or method == 'center_and_resize_2d_xz':
            target_shape = (target_shape[0], target_shape[2]) if len(target_shape) == 3 else (target_shape[0], target_shape[1])
        elif method == 'interpol_2d_yz' or method == 'zero_pad_2d_yz' or method == 'center_and_resize_2d_yz':
            target_shape = (target_shape[1], target_shape[2]) if len(target_shape) == 3 else (target_shape[0], target_shape[1])

    # Apply the chosen method
    if method == 'interpol_3d':
        processed_t1_data = interpolate_3d(t1_data, target_shape)
        processed_mask_data = interpolate_3d(mask_data, target_shape)
    elif method == 'interpol_2d_xy':
        processed_t1_data = interpolate_2d_entire_vol(t1_data, target_shape, direction = 'z')
        processed_mask_data = interpolate_2d_entire_vol(mask_data, target_shape, direction = 'z')
    elif method == 'interpol_2d_xz':
        processed_t1_data = interpolate_2d_entire_vol(t1_data, target_shape, direction = 'y')
        processed_mask_data = interpolate_2d_entire_vol(mask_data, target_shape, direction = 'y')
    elif method == 'interpol_2d_yz':
        processed_t1_data = interpolate_2d_entire_vol(t1_data, target_shape, direction = 'x')
        processed_mask_data = interpolate_2d_entire_vol(mask_data, target_shape, direction = 'x')

    elif method == 'zero_pad_3d':
        processed_t1_data = zero_pad_3d(t1_data, target_shape)
        processed_mask_data = zero_pad_3d(mask_data, target_shape)
    elif method == 'zero_pad_2d_xy':
        processed_t1_data = zero_pad_2d_entire_vol(t1_data, target_shape, direction = 'z')
        processed_mask_data = zero_pad_2d_entire_vol(mask_data, target_shape, direction = 'z')
    elif method == 'zero_pad_2d_xz':
        processed_t1_data = zero_pad_2d_entire_vol(t1_data, target_shape, direction = 'y')
        processed_mask_data = zero_pad_2d_entire_vol(mask_data, target_shape, direction = 'y')
    elif method == 'zero_pad_2d_yz':
        processed_t1_data = zero_pad_2d_entire_vol(t1_data, target_shape, direction = 'x')
        processed_mask_data = zero_pad_2d_entire_vol(mask_data, target_shape, direction = 'x')

    elif method == 'center_and_resize_3d':
        processed_t1_data = center_and_resize_3d(t1_data, target_shape)
        processed_mask_data = center_and_resize_3d(mask_data, target_shape)
    elif method == 'center_and_resize_2d_xy':
        processed_t1_data = center_and_resize_2d(t1_data, target_shape, direction = 'z')
        processed_mask_data = center_and_resize_2d(mask_data, target_shape, direction = 'z')
    elif method == 'center_and_resize_2d_xz':
        processed_t1_data = center_and_resize_2d(t1_data, target_shape, direction = 'y')
        processed_mask_data = center_and_resize_2d(mask_data, target_shape, direction = 'y')
    elif method == 'center_and_resize_2d_yz':
        processed_t1_data = center_and_resize_2d(t1_data, target_shape, direction = 'x')
        processed_mask_data = center_and_resize_2d(mask_data, target_shape, direction = 'x')
    else:
        raise ValueError("Invalid method. Use one of: 'interpol_3d', 'interpol_2d_xy', 'interpol_2d_xz', 'interpol_2d_yz', 'zero_pad_3d', 'zero_pad_2d_xy', 'zero_pad_2d_xz', 'zero_pad_2d_yz.")
    
    root_path = os.path.abspath(os.path.join(data_path, os.pardir))
    # Create new folder for processed data
    new_data_path = os.path.join(root_path, 'ANON_DATA_'+f'{method.upper()}_'+'_'.join(str(x) for x in target_shape))
    print("new data path: " + new_data_path)
    os.makedirs(new_data_path, exist_ok=True)
    
    # postprocess mask data to just have 0 and 1 as labels
    processed_mask_data[processed_mask_data >= 0.5] = 1
    processed_mask_data[processed_mask_data < 0.5] = 0

    # Create output folders
    new_subject_folder = os.path.join(new_data_path, subject_name)
    os.makedirs(new_subject_folder, exist_ok=True)

    # Save processed T1.nii and mask.nii
    if save_data:
        output_t1_file = os.path.join(new_subject_folder, 'T1.nii')
        output_mask_file = os.path.join(new_subject_folder, 'mask.nii')
        print('saving to ' + new_subject_folder)
        nib.save(nib.Nifti1Image(processed_t1_data, affine=None), output_t1_file)
        nib.save(nib.Nifti1Image(processed_mask_data, affine=None), output_mask_file)


def process_all_subjects(data_path, method, target_shape = (256, 256, 256), save_data = True):
    # Process each subject in the root path
    for subject_name in sorted(os.listdir(data_path)):
        print("processing subject: " + subject_name)
        if subject_name in ['.DS_Store','README.md']:
            continue
        process_subject(data_path, subject_name, method, target_shape, save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--root_path', help='Root path where to store the data (project folder)', required=False, default="/Users/liaschmid/Documents/Uni Heidelberg/7. Semester Thesis/project")
    parser.add_argument('-i', '--interpolation_method', help='Which method to use for interpolation, one of "interpol_2d_xy", "zero_pad_2d_xz", "zero_pad_2d_yz",  "interpol_3d", "zero_pad_3d" ', required=False, default="interpol_3d")
    parser.add_argument('-shape', '--target_shape', help='Target shape for interpolation', required=True, default="(256, 256, 256)")
    parser.add_argument('-s', '--save_data', help='Whether to save the data', required=False, default=True)
    args = parser.parse_args()

    root_path = args.root_path
    data_path = args.root_path + "/ANON_DATA"
    target_shape = eval(args.target_shape)
    print("root path: " + args.root_path)
    print("interpolation method: " + args.interpolation_method)
    print("target shape and type: " + args.target_shape + " " + str(type(target_shape)))
    print("save data: " + str(args.save_data))
    
    process_all_subjects(data_path, args.interpolation_method, target_shape, args.save_data)
