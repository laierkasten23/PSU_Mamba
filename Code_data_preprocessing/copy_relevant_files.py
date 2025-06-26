import os
import shutil

# Define the source and destination directories
source_dir = '/home/linuxuser/user/data/pazienti81_end'
dest_dir = '/home/linuxuser/user/data/pazienti_final_tests'

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# List of files to copy
files_to_copy = [
    "_ChP_mask_FLAIR_manual_seg.nii",
    "_ChP_mask_T1_manual_seg.nii",
    "_ChP_mask_T1xFLAIR_manual_seg.nii",
    "_T1.nii",
    "_FLAIR.nii",
    "_T1xFLAIR.nii"
]

# Iterate over the directories in the source directory
for subject_id in os.listdir(source_dir):
    subject_source_dir = os.path.join(source_dir, subject_id)
    subject_dest_dir = os.path.join(dest_dir, subject_id)
    
    # Ensure the subject destination directory exists
    os.makedirs(subject_dest_dir, exist_ok=True)
    
    # Copy the specified files
    for file_suffix in files_to_copy:
        source_file = os.path.join(subject_source_dir, f"{subject_id}{file_suffix}")
        if os.path.isfile(source_file):
            dest_file = os.path.join(subject_dest_dir, f"{subject_id}{file_suffix}")
            shutil.copy2(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}")

print("Files copied successfully.")