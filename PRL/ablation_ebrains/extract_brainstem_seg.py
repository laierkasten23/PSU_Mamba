# in /data1/LIA/BIDS, loop throught all subjects sub-00x/seg read the sub-00x_ses-00x_training_labels.nii.gz file. Create a new file sub-00x_ses-00x_training_labels_brainstem.nii.gz with only the brainstem segmentation which corresponds to the label 6 in the original file. 
# Save with same metadata as the original file. Save it to the same directory as the original file.

import nibabel as nib
import os
import numpy as np
def extract_brainstem_segmentation(dataset_dir):
    """
    Extracts brainstem segmentation from the training labels of each subject in the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory containing subject folders.
    """
    for subject in os.listdir(dataset_dir):
        subject_dir = os.path.join(dataset_dir, subject)
        if os.path.isdir(subject_dir):
            seg_file = os.path.join(subject_dir, "seg", f"{subject}_ses-001_training_labels.nii.gz")
            print(f"Processing {seg_file}")
            if os.path.exists(seg_file):
                # Load the segmentation file
                seg_img = nib.load(seg_file)
                seg_data = seg_img.get_fdata()
                
                # Create a new array for brainstem segmentation
                brainstem_segmentation = np.zeros_like(seg_data)
                
                # Extract brainstem (label 6)
                brainstem_segmentation[seg_data == 6] = 6
                
                # change it to 1
                brainstem_segmentation[brainstem_segmentation == 6] = 1
                
                # Create a new NIfTI image
                new_seg_img = nib.Nifti1Image(brainstem_segmentation, seg_img.affine, seg_img.header)
                
                # Save the new segmentation file
                new_seg_file = os.path.join(subject_dir, "seg", f"{subject}_ses-001_training_labels_brainstem.nii.gz")
                nib.save(new_seg_img, new_seg_file)
                print(f"Saved brainstem segmentation for {subject} to {new_seg_file}")
                
                
if __name__ == "__main__":
    dataset_dir = '/data1/LIA/BIDS'  # Change this to your dataset directory
    extract_brainstem_segmentation(dataset_dir)
# This script will loop through all subjects in the specified dataset directory, extract the brainstem segmentation
