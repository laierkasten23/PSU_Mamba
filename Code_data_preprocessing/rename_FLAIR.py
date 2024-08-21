import os
import re

def rename_old_flair_files(base_dir):
    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Find the files in the subfolder
            for filename in sorted(os.listdir(subfolder_path)):
                if re.match(r'^\d{3}_FLAIR\.nii$', filename):
                    old_filename = filename
                    new_filename = f"old_{filename}"
                    os.rename(os.path.join(subfolder_path, old_filename), os.path.join(subfolder_path, new_filename))
                    print(f"Renamed {old_filename} to {new_filename} in {subfolder_path}")

def rename_r_flair_files(base_dir):
    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Find the files in the subfolder
            for filename in sorted(os.listdir(subfolder_path)):
                if re.match(r'^r\d{3}_FLAIR\.nii$', filename):
                    old_filename = filename
                    new_filename = filename[1:]  # Remove the leading 'r'
                    os.rename(os.path.join(subfolder_path, old_filename), os.path.join(subfolder_path, new_filename))
                    print(f"Renamed {old_filename} to {new_filename} in {subfolder_path}")

def rename_chp_mask_files(base_dir):
    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Find the files in the subfolder
            for filename in sorted(os.listdir(subfolder_path)):
                if re.match(r'^\d{3}_ChP_mask_FLAIR_manual_seg\.nii$', filename):
                    old_filename = filename
                    new_filename = f"o_{filename}"
                    os.rename(os.path.join(subfolder_path, old_filename), os.path.join(subfolder_path, new_filename))
                    print(f"Renamed {old_filename} to {new_filename} in {subfolder_path}")


if __name__ == "__main__":
    base_dir = "/home/linuxlia/Lia_Masterthesis/data/pazienti"  # Insert your data directory here
    #rename_old_flair_files(base_dir)
    #rename_r_flair_files(base_dir)
    rename_chp_mask_files(base_dir)