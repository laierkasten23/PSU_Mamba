import argparse
import os
import random
import shutil
import sys
import nibabel as nib

def convert_nii_to_niigz(input_filepath, output_filepath):
    try:
        img = nib.load(input_filepath)
        nib.save(img, output_filepath)
        print(f"Converted {input_filepath} to {output_filepath}")
    except Exception as e:
        print(f"Error converting {input_filepath} to {output_filepath}: {e}")


def anonymize_data(path, new_folder_name):
    """
    This function anonymizes the dataset by renaming the files and folders to a new folder name
    :param path: path to the dataset
    :param new_folder_name: new name of the folder
    :return:

    This function creates a new folder from the folder structure:
    DATASET
        -Name_1
            -ChP_mask_user_Name_1.nii.gz
            -ChP_mask_user_Name_1.nii
            -coFLAIR_3D_301_Name_1.nii
            -...
        -Name_2
            -ChP_mask_user_Name_2.nii.gz
            -ChP_mask_user_Name_2.nii
            -coFLAIR_3D_301_Name_2.nii
            -...
        ...
    where Name_j is the subjects name. In the new folder just ChP_mask_user_Name_1.nii and coFLAIR_3D_301_Name_1.nii are needed 
    (for the mask and the image respectively).
    to the following structure:
    
    DATASET
    |-01
        |-mask.nii
        |-coFLAIR.nii 
    |-02
        |-mask.nii
        |-coFLAIR.nii
    ...
    """
    # Create the folder structure
    new_dataset_path = os.path.join(path, '..', new_folder_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    print('Folder structure created')

    # Get the list of all the subjects
    subjects = sorted(os.listdir(path))
    print(subjects)

    # if '.DS_Store', 'flair_coregistration.m' or 'origin_setting.txt' is 'in subjects' remove it
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')
    if 'flair_coregistration.m' in subjects:
        subjects.remove('flair_coregistration.m')
    if 'origin_setting.txt' in subjects:
        subjects.remove('origin_setting.txt')


    # Copy the images and labels to the correct folder
    for (i, subject) in enumerate(subjects):
        print(f"{i:02d}", subject)
    
    
        # Create the subject folder
        subject_folder = os.path.join(new_dataset_path, str(i).zfill(2))
        print(subject_folder)
        
        os.makedirs(subject_folder, exist_ok=True)

        # Copy the images and labels and substitute the names by numbers to anonymize the dataset

        for file in sorted(os.listdir(os.path.join(path, subject))):
            '''
            if 'mask' in file and not 'gz' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, 'mask.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, 'mask.nii')))
            if 'co_FLAIR' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, 'co_FLAIR.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, 'co_FLAIR.nii')))
            if 'T1xFLAIR_' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_T1xFLAIR.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_T1xFLAIR.nii')))
            if 'coreg_cs_T1W_3D_TFE_mdc_' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_T1_mdc.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_T1_mdc.nii')))
            '''
            if 'T1xFLAIR' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_ChP_seg_T1xFLAIR.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_ChP_seg_T1xFLAIR.nii')))
            if 'T1mdc' in file:
                shutil.copy(os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_ChP_seg_T1mdc.nii'))
                print('Copying %s to %s' % (os.path.join(path, subject, file), os.path.join(subject_folder, f'{str(i).zfill(2)}_ChP_seg_T1mdc.nii')))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anonymize the dataset')
    parser.add_argument('--path', type=str, help='Path to the dataset', default='/Users/user/Documents/projects_/project_dir/pazienti_flair_coreg')
    parser.add_argument('--new_folder_name', type=str, help='ANON_FLAIR_COREG', default='ANON_FLAIR_COREG')
    args = parser.parse_args()
    anonymize_data(args.path, args.new_folder_name)


'''
    python3 step0_anonymize_sym_dataset.py --path '/var/datasets/user/_preanalysis_data/processed_data_Dem_SM' --new_folder_name 'ANON__preanalysis'
    python3 step0_anonymize_sym_dataset.py --path '/data1/user/processed_data_Dem_SM' --new_folder_name 'ANON__preanalysis_Dem_SM'
    
    '''
