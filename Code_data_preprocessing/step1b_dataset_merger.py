import os
import nibabel as nib


def dataset_merger(dataset_path_list, new_dataset_path):
    """
    Merge multiple datasets into one dataset

    Args:
        dataset_path_list: list of paths to the datasets that need to be merged
        new_dataset_path: path to the new dataset

    All dataset_path_list should have the following structure:
    DATASET
        -image_Tr
            -MRI_IDsj_image.nii
            - ....
        -image_Ts
            -MRI_IDsj_image.nii
            - ....
        -label_Tr
            -MRI_IDsj_seg.nii
            - ....
        
    It might be possible that in one folder there is only the image_Ts folder, no other folder. 
    The script goes through all the folders and copies the files to the new dataset folder and renames them such that there is a continuous numbering of the files.

    In the end the new dataset folder will have the following structure:
    DATASET
        -image_Tr
            -MRI_IDsj_image.nii
            - ....
        -image_Ts
            -MRI_IDsj_image.nii
            - ....
        -label_Tr
            -MRI_IDsj_seg.nii
            - ....
    having merged all the datasets in the dataset_path_list.

    """
    # Create the new dataset folder
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Create the image and label folders
    if not os.path.exists(os.path.join(new_dataset_path, 'image_Tr')):
        os.makedirs(os.path.join(new_dataset_path, 'image_Tr'))
    if not os.path.exists(os.path.join(new_dataset_path, 'label_Tr')):
        os.makedirs(os.path.join(new_dataset_path, 'label_Tr'))
    if not os.path.exists(os.path.join(new_dataset_path, 'image_Ts')):
        os.makedirs(os.path.join(new_dataset_path, 'image_Ts'))


    # Go through all the datasets
    


