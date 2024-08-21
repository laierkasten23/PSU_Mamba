import nibabel as nib
import numpy as np
import glob
import argparse
import os


# function to read in a nifti file and return the affine matrix
#

def compute_affine_matrix(nifti_file):
    """
    Given a path to a nifti file, read the file
    and return the affine matrix.

    Args:
    - nifti_file (str): Path to the nifti file.

    Returns:
    - numpy.ndarray: The affine matrix.
    """
    # Load the nifti file
    nifti = nib.load(nifti_file)
    # Return the affine matrix
    return nifti.affine

def compare_affine_matrices(nifti_file1, nifti_file2):
    """
    Given two paths to nifti files, read the files
    and compare the affine matrices.

    Args:
    - nifti_file1 (str): Path to the first nifti file.
    - nifti_file2 (str): Path to the second nifti file.

    Returns:
    - bool: True if the affine matrices are equal, False otherwise.
    """
    # Load the nifti files
    nifti1 = nib.load(nifti_file1)
    nifti2 = nib.load(nifti_file2)
    # Compare the affine matrices
    #print("nifti1.affine = ", nifti1.affine)
    #print("nifti2.affine = ", nifti2.affine)
    return np.allclose(nifti1.affine, nifti2.affine), nifti1.affine, nifti2.affine

def filter_valid_images(image_paths, image_type):
    """
    Filter out invalid image paths that do not match the pattern XXX_<image_type>.nii.

    Args:
    - image_paths (list): List of image paths.
    - image_type (str): The type of image to filter for (e.g., 'FLAIR', 'T1').

    Returns:
    - list: List of valid image paths.
    """
    valid_images = []
    for path in image_paths:
        filename = os.path.basename(path)
        if filename.count('_') == 1 and filename.endswith(f'_{image_type}.nii'):
            valid_images.append(path)
    return valid_images

def get_nifti_by_modality(root_dir, image_type):
    """
    Given a root directory, find all files ending with the specified image type.

    Args:
    - root_dir (str): Path to the root directory containing subject folders.
    - image_type (str): The type of image to search for (e.g., 'FLAIR', 'T1').

    Returns:
    - list: List of paths to the specified type of images.
    """
    pattern = f'*_{image_type}.nii'
    image_paths = glob.glob(os.path.join(root_dir, '*', pattern))
    return [path for path in image_paths if os.path.basename(path).count('_') == 1] 

root_dir = '/home/linuxlia/Lia_Masterthesis/data/pazienti'
image_type = 'FLAIR'  # Change to 'T1' or other types as needed
flair_images = sorted(get_nifti_by_modality(root_dir, image_type))
t1_images = sorted(get_nifti_by_modality(root_dir, 'T1'))

print("flair_images = ", flair_images)
print("t1_images = ", t1_images)

#affine_flair1 = compute_affine_matrix(flair_images[26])
#affine_t11 = compute_affine_matrix(t1_images[26])
#print("affine_flair27 = ", flair_images[26], affine_flair1)
#print("affine_t1104 = ", t1_images[26], affine_t11)

list_comparing_flair_vs_t1 = []
list_flair_affine = []
list_t1_affine = []
for i in range(len(flair_images)):
    checker_flair_vs_t1, affine_flair, affine_t1 = compare_affine_matrices(flair_images[i], t1_images[i])
    list_flair_affine.append(affine_flair)
    list_t1_affine.append(affine_t1)
    if checker_flair_vs_t1:
        list_comparing_flair_vs_t1.append(str(i+1) + " check")
    else:
        list_comparing_flair_vs_t1.append(str(i+1) + " affine matrix NOT equal")

#flairvst1_104 = compare_affine_matrices(flair_images[26], t1_images[26])
#flairvst1_1 = compare_affine_matrices(flair_images[0], t1_images[0])

#print("flair1vst11 = ", flairvst1_104)

#print(list_comparing_flair_vs_t1)

#t1_38_path = '/home/linuxlia/Lia_Masterthesis/data/pazienti1_52_Lia/038/038_T1.nii' 
#flair_38_path = '/home/linuxlia/Lia_Masterthesis/data/pazienti1_52_Lia/038/038_FLAIR.nii'
#flair_38_coregistred_path = '/home/linuxlia/Lia_Masterthesis/data/pazienti1_52_Lia/038/038_FLAIR_Lia_registered.nii'

#t1_vs_flair_38 = compare_affine_matrices(t1_38_path, flair_38_path)
#t1_vs_flair_38_coregistred = compare_affine_matrices(t1_38_path, flair_38_coregistred_path)
#print("t1_vs_flair_38 = ", t1_vs_flair_38)
#print("t1_vs_flair_38_coregistred = ", t1_vs_flair_38_coregistred)
print(list_comparing_flair_vs_t1)
print(list_flair_affine[87] - list_t1_affine[87])
