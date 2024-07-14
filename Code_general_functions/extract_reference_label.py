import os
import re


# The files still need to have three digits in the file name, otherwise atm it does not work. 
# TODO: Change the code to work with any number of digits in the file name.

def get_reference_label_path(image_path, reference_labels_base_path):
    """
    Given an image path and the base path for reference labels,
    return the path to the corresponding reference label.
    
    Args:
    - image_path (str): The path to the current image or label.
    - reference_labels_base_path (str): The base path where reference labels are stored.
    
    Returns:
    - str: Path to the corresponding reference label.
    """
    # Extract the index from the image filename
    # Assuming filenames are in the format "index_description.ext" or similar
    index_match = re.search(r'(\d+)', os.path.basename(image_path))
    print(index_match)
    if not index_match:
        raise ValueError("Could not extract index from image path.")
    
    index = index_match.group(1)
    print(index)
    
    # Construct the pathname for the reference label
    # Assuming reference labels have a specific naming pattern you can adjust below
    reference_label_filename = f"{index}_ChP_mask_T1xFLAIR_manual_seg.nii"  # T1xFLAIR reference label
    # Walk through the directory tree to find the file

    for dirpath, dirnames, filenames in os.walk(reference_labels_base_path):
        if reference_label_filename in filenames:
            return os.path.join(dirpath, reference_label_filename)
    
    raise FileNotFoundError(f"Reference label {reference_label_filename} not found in {reference_labels_base_path} or its subdirectories. SKIPPING!!")


# Example usage
#image_path = "/home/linuxlia/Lia_Masterthesis/data/Dataset009_ChoroidPlexus_T1_sym_AP/image_Tr/010_image0001.nii"
#image_path = "/home/linuxlia/Lia_Masterthesis/data/Dataset009_ChoroidPlexus_T1_sym_UMAMBA/imagesTs/026_image0001.nii"
#reference_labels_base_path = "/home/linuxlia/Lia_Masterthesis/data/reference_labels/ref_labelTs"
#reference_label_path = get_reference_label_path(image_path, reference_labels_base_path)
#print(reference_label_path)


def get_reference_label_paths(image_paths, reference_labels_base_path):
    """
    Given a list of image paths and the base path for reference labels,
    return a list of paths to the corresponding reference labels.
    
    Args:
    - image_paths (list of str): The paths to the current images or labels.
    - reference_labels_base_path (str): The base path where reference labels are stored.
    
    Returns:
    - list of str: Paths to the corresponding reference labels.
    """
    reference_label_paths = []
    for image_path in image_paths:
        try:
            reference_label_path = get_reference_label_path(image_path, reference_labels_base_path)
            reference_label_paths.append(reference_label_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            

            
    return reference_label_paths

# Example usage
#image_paths = [
#    "/home/linuxlia/Lia_Masterthesis/data/Dataset009_ChoroidPlexus_T1_sym_AP/image_Tr/010_image0001.nii",
#    "/home/linuxlia/Lia_Masterthesis/data/Dataset009_ChoroidPlexus_T1_sym_UMAMBA/imagesTs/025_image0001.nii"
#]
#reference_labels_base_path = "/home/linuxlia/Lia_Masterthesis/data/reference_labels/ref_labelTr"
#reference_label_paths = get_reference_label_paths(image_paths, reference_labels_base_path)
#for path in reference_label_paths:
#    print(path)