import os
import shutil


dividedbysubject = False

# Define the source and destination directories

if dividedbysubject:
    source_dir = '/home/linuxlia/Lia_Masterthesis/data/pazienti'
    dest_dir = '/home/linuxlia/Lia_Masterthesis/data/pazienti_tobecontrolled_T1xFLAIR'
else:
    source_dir = '/home/linuxlia/Lia_Masterthesis/data/T1xFLAIR_mask_ref'
    dest_dir = '/home/linuxlia/Lia_Masterthesis/data/pazienti'


if dividedbysubject:
    print(f"Copying from {source_dir} to {dest_dir}")
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over the directories in the source directory
    for patient_dir in os.listdir(source_dir):
        # Construct the full path to the source file
        source_file = os.path.join(source_dir, patient_dir, f"{patient_dir}_ChP_mask_T1xFLAIR_manual_seg.nii")
        
        # Check if the source file exists
        if os.path.isfile(source_file):
            # Construct the full path to the destination file
            dest_file = os.path.join(dest_dir, f"{patient_dir}_ChP_mask_T1xFLAIR_manual_seg.nii")
            
            # Copy the source file to the destination file
            shutil.copy2(source_file, dest_file)

else:
    print(f"Copying from {source_dir} to {dest_dir}")
    # Iterate over the files in the source directory
    for filename in os.listdir(source_dir):
        # Construct the full path to the source file
        source_file = os.path.join(source_dir, filename)

        # Extract the patient directory name from the filename
        patient_dir = filename.split('_')[0]  # Assuming the filename format is "xxx_ChP_mask_T1xFLAIR_manual_seg.nii"
        
        # Construct the path to the destination subdirectory
        dest_subdir = os.path.join(dest_dir, patient_dir)
        
        # Ensure the destination subdirectory exists
        os.makedirs(dest_subdir, exist_ok=True)
        
        # Construct the full path to the destination file
        dest_file = os.path.join(dest_subdir, filename)
        
        # Copy the source file to the destination file
        shutil.copy2(source_file, dest_file)