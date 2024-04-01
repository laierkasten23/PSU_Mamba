import argparse
import os
import random
import shutil
import nibabel as nib

def folderstructure_changer(path, folder_name, amount_train_subjects='all'):
    """
    This function changes the folder structure of the dataset
    from the following structure:
    DATASET
        -MRI_IDsj
            -T1.nii
            -mask.nii
        -MRI_IDsj
            -T1.nii
            -mask.nii
        ...
    where MRI_IDsj is the subject ID from 0 to N 
    to the following structure:
    
    DATASET
    |-MRI_IDsj_image.nii.gz
    |-MRI_IDsi_image.nii.gz
    |- ....
    |-labels
        |-final
            |-MRI_IDsj_seg.nii.gz
            |- ....

    where i,j are the subject IDs from 0 to N, the folder labels contains a subfolder final containing the 
    segmentation masks of the corresponding amount_train_subjects images which were randomly chosen. 

    :param path: path to the dataset
    :param folder_name: name of the folder to be created
    :param amount_train_subjects: amount of subjects to be used for training (meaning, the labels are known for these subjects). 
        If 'all' is given, all subjects are used for training, otherwise integer value is expected


    Example Usage: 
    python step1a_create_monai_dataset.py --path /var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels --folder_name dataset_monai --amount_train_subjects 10
    python step1a_create_monai_dataset.py --path /var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels --folder_name dataset_monai_train_from_scract --amount_train_subjects 'all'
    """


    if not (amount_train_subjects == 'all') and not isinstance(amount_train_subjects, int):
        print("entered", amount_train_subjects)
        print("not (amount_train_subjects == 'all')", not (amount_train_subjects == 'all'))
        print(amount_train_subjects is not 'all')
        raise ValueError('amount_train_subjects has to be either "all" or an integer')
    
    # Create the folder structure
    new_dataset_path = os.path.join(path, '..', folder_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'labels', 'final'), exist_ok=True)
    print('Folder structure created')
    

    # Get the list of all the subjects
    subjects = os.listdir(path)
    #subjects.remove(folder_name)
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')

    # Randomly choose the train subjects and test subjects
    train_subjects = [subject for subject in subjects]

    if amount_train_subjects == 'all':
        test_subjects = [subject for subject in subjects]
    else:
        test_subjects = random.sample(subjects, amount_train_subjects)
    print('Train subjects: %s' % train_subjects)
    print('Test subjects: %s' % test_subjects)
                
    # Copy the images and labels to the correct folder
    for subject in train_subjects:
        img = nib.load(os.path.join(path, subject, 'T1.nii'))
        nib.save(img, os.path.join(new_dataset_path, subject + '_ChP.nii.gz'))
        
    for subject in test_subjects:
        mask = nib.load(os.path.join(path, subject, 'mask.nii'))
        nib.save(mask, os.path.join(new_dataset_path, 'labels', 'final', subject + '_ChP.nii.gz'))
        
    print('Images and labels copied to the correct folder')
    
    return new_dataset_path


def folderstructure_changer_reverse(path, folder_name, amount_train_subjects='all'):
    """
    This function changes the folder structure of the dataset from the following structure 
    
    DATASET
    |-MRI_IDsj_image.nii.gz
    |-MRI_IDsi_image.nii.gz
    |- ....
    |-labels
        |-final
            |-MRI_IDsj_seg.nii.gz
            |- ....

    where i,j are the subject IDs from 0 to N, the folder labels contains a subfolder final containing the 
    segmentation masks of the corresponding amount_train_subjects images which were randomly chosen. 


    to the following structure:
    DATASET
        -MRI_IDsj
            -T1.nii
            -mask.nii
        -MRI_IDsj
            -T1.nii
            -mask.nii
        ...
    where MRI_IDsj is the subject ID from 0 to N

    :param path: path to the dataset
    :param folder_name: name of the folder to be created
    :param amount_train_subjects: amount of subjects to be used for training (meaning, the labels are known for these subjects). 
        If 'all' is given, all subjects are used for training, otherwise integer value is expected
    
    """

    if not (amount_train_subjects == 'all') and not isinstance(amount_train_subjects, int):
        print("entered", amount_train_subjects)
        print("not (amount_train_subjects == 'all')", not (amount_train_subjects == 'all'))
        print(amount_train_subjects is not 'all')
        raise ValueError('amount_train_subjects has to be either "all" or an integer')
    
    # Create the folder structure
    new_dataset_path = os.path.join(path, '..', folder_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    print('Folder structure created')
    

    # Get the list of all the subjects
    subjects = os.listdir(os.path.join(path, 'labels', 'final'))
    #subjects.remove(folder_name)
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')

    # Randomly choose the train subjects and test subjects
    train_subjects = [subject for subject in subjects]

    if amount_train_subjects == 'all':
        test_subjects = [subject for subject in subjects]
    else:
        test_subjects = random.sample(subjects, amount_train_subjects)
    print('Train subjects: %s' % train_subjects)
    print('Test subjects: %s' % test_subjects)
                
    # Copy the images and labels to the correct folder
    for subject in train_subjects:
        img = nib.load(os.path.join(path, 'labels', 'final', subject + '_ChP.nii.gz'))
        nib.save(img, os.path.join(new_dataset_path, subject, 'mask.nii'))
        
    for subject in test_subjects:
        mask = nib.load(os.path.join(path, subject + '_ChP.nii.gz'))
        nib.save(mask, os.path.join(new_dataset_path, subject, 'T1.nii'))
        
    print('Images and labels copied to the correct folder')
    
    return new_dataset_path



if __name__ == '__main__':
    print('Starting reorganizing Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels')
    parser.add_argument('--folder_name', type=str, default='dataset_monai', help="name of the folder to be created")
    parser.add_argument('--amount_train_subjects', default='all', help="If 'all' is given, all subjects are used for training, otherwise integer value is expected which determines the amount of subjects used for training as labels are known for these subjects")
    args = parser.parse_args()

    folderstructure_changer(args.path, args.folder_name, args.amount_train_subjects)
    print('Reorganizing Dataset finished')
    