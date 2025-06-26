import os
import nibabel as nib
import numpy as np

def check_affine_matrices(parent_dir):
    # List of file suffixes to check
    files_to_check = [
        "_ChP_mask_FLAIR_manual_seg.nii",
        "_ChP_mask_T1_manual_seg.nii",
        "_ChP_mask_T1xFLAIR_manual_seg.nii",
        "_FLAIR.nii",
        "_T1xFLAIR.nii"
    ]

    # Initialize dictionaries to store results and affine matrices
    results = {suffix: [] for suffix in files_to_check}
    affine_matrices = {suffix: {} for suffix in files_to_check}


    # Iterate over the directories in the parent directory
    for subject_id in os.listdir(parent_dir):
        subject_dir = os.path.join(parent_dir, subject_id)
        
        # Load the reference T1.nii file
        reference_file = os.path.join(subject_dir, f"{subject_id}_T1.nii")
        if not os.path.isfile(reference_file):
            print(f"Reference file {reference_file} not found for subject {subject_id}")
            continue
        
        reference_img = nib.load(reference_file)
        reference_affine = reference_img.affine
        
        # Check the affine matrices of the other files
        for file_suffix in files_to_check:
            file_to_check = os.path.join(subject_dir, f"{subject_id}{file_suffix}")
            if not os.path.isfile(file_to_check):
                print(f"File {file_to_check} not found for subject {subject_id}")
                continue
            
            img_to_check = nib.load(file_to_check)
            affine_to_check = img_to_check.affine

            affine_matrices[file_suffix][subject_id] = affine_to_check
            
            
            if np.allclose(reference_affine, affine_to_check):
                results[file_suffix].append((subject_id, "check"))
            else:
                results[file_suffix].append((subject_id, "matrices are not equal"))

    # Print the results sorted by subject identifier
    for file_suffix, result_list in results.items():
        print(f"Results for {file_suffix}:")
        for subject_id, status in sorted(result_list):
            print(f"{subject_id}: {status}")
            #print(f"Affine matrix:\n{affine_matrices[file_suffix][subject_id]}")

    return results, affine_matrices


# Example usage
parent_directory = '/home/linuxuser/user/data/pazienti_final_tests'
check_affine_matrices(parent_directory)