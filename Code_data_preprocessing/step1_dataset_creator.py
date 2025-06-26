import argparse
import os
import random
from random import sample
import shutil
import nibabel as nib
from typing import Union, List

def folderstructure_changer(path, 
                            task_id = 53,
                            task_name = "Choroid_Plexus",
                            test_data_only = False, 
                            amount_train_subjects:int =5, 
                            train_val_split:float = 0.8,
                            train_test_index_list: Union[List, None] = None, 
                            datasettype:str ='ASCHOPLEX', 
                            modality:str ='T1', 
                            add_id_img:str = 'image', 
                            add_id_lab:str = 'seg'):
    """
    This function changes the folder structure of the datasets
    from the following structure:
    DATASET
        -MRI_IDsj
            -T1.nii / FLAIR.nii / T2.nii / T1gd.nii / image.nii
            -mask.nii / seg.nii
        -MRI_IDsj
            -T1.nii / FLAIR.nii / T2.nii / T1gd.nii / image.nii
            -mask.nii / seg.nii
        ...
    where MRI_IDsj is the subject ID from 000 to N 
    to the following structure:

    if datasettype = ASCHOPLEX:
    
    DATASET
    |-image_Tr
        |-MRI_IDsj_image.nii
        |- ....
    |-image_Ts
        |-MRI_IDsj_image.nii
        |- ....
    |-label_Tr
        |-MRI_IDsj_seg.nii
        |- ....

    if datasettype = NNUNETV2:
    
    Dataset{IDENTIFER}_ChoroidPlexus
    |-imagesTr
        |-MRI_IDsj_{XXXX}.nii
        |-MRI_IDsj_{XXXY}.nii
        |- ....
    |-imagesTs  (optional)
        |-MRI_IDsj_{XXXX}.nii
        |-MRI_IDsj_{XXXY}.nii
        |- ....
    |-labelsTr
        |-MRI_IDsj.nii
        |- ...

    if datasettype = 'UMAMBA':
    
    Dataset{IDENTIFER}_ChoroidPlexus
    |-imagesTr
        |-MRI_IDsj_{XXXX}.nii.gz
        |-MRI_IDsj_{XXXY}.nii.gz
        |- ....
    |-imagesVal
        |-MRI_IDsj_{XXXX}.nii.gz
        |-MRI_IDsj_{XXXY}.nii.gz
        |- ....
    |-imagesTs  (optional)
        |-MRI_IDsj_{XXXX}.nii.gz
        |-MRI_IDsj_{XXXY}.nii.gz
        |- ....
    |-labelsTr
        |-MRI_IDsj.nii.gz
        |- ...
    |-labelsVal
        |-MRI_IDsj.nii.gz
        |- ...
    
    where images_Tr/ imagesTr is the folder containing the training images, 
     imagesVal is the folder containing the validation images
       and images_Ts /imagesTs is the folder containing the test images 
    from which amount_test_subjects are randomly chosen from the original dataset.
    In case of datasettype = NNUNETV2, the images are named MRI_IDsj_{XXXX}.nii and the labels are named MRI_IDsj.nii, where XXXX is the 4-digit modality/channel identifier 
    - FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003).
    - One id can have multiple modalities, e.g. MRI_IDsj_{XXXX}.nii, MRI_IDsj_{XXXY}.nii, MRI_IDsj_{XXYZ}.nii,...}.nii

    :param path: path to the dataset
    :param task_id: id of the task
    :param task_name: name of the task
    :param test_data_only: if True, only the test data is created and masks do not exist
    :param amount_train_subjects: amount of subjects to be used for training 
    :param datasettype: type of the dataset, either ASCHOPLEX or NNUNETV2 or UMAMBA
    :param modality: modality of the images, either T1, T2, FLAIR, T1gd. Can be a list of modalities.
    :param add_id_img: additional identifier for the image files, default is 'image'. Could be ''. 
    :param add_id_lab: additional identifier for the label files, default is 'seg'. Could be ''.
    
    :return:
    
    """

    # Create the folder structure
    folder_name = "Dataset%03.0d_%s" % (task_id, task_name)

    new_dataset_path = os.path.join(path, '..', folder_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    if datasettype == 'ASCHOPLEX':
        train_folder_name = 'image_Tr'
        test_folder_name = 'image_Ts'
        label_folder_name = 'label_Tr'
    elif datasettype == 'NNUNETV2':
        train_folder_name = 'imagesTr'
        test_folder_name = 'imagesTs'
        label_folder_name = 'labelsTr'
    elif datasettype == 'UMAMBA':
        train_folder_name = 'imagesTr'
        val_folder_name = 'imagesVal'
        test_folder_name = 'imagesTs'
        label_folder_name = 'labelsTr'
        label_val_folder_name = 'labelsVal'

    if modality == 'FLAIR':
        identifier = '0000'
    elif modality == 'T1':
        identifier = '0001'
    elif modality == 'T1gd':
        identifier = '0002'
    elif modality == 'T2':
        identifier = '0003'
    else:
        raise ValueError('Modality not supported')

    if not test_data_only:
        os.makedirs(os.path.join(new_dataset_path, train_folder_name), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, label_folder_name), exist_ok=True)
    
    os.makedirs(os.path.join(new_dataset_path, test_folder_name), exist_ok=True)

    if datasettype == 'UMAMBA':
        os.makedirs(os.path.join(new_dataset_path, val_folder_name), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, label_val_folder_name), exist_ok=True)

    print('Folder structure created at %s' % new_dataset_path)
    
    # Get the list of all the subjects
    subjects = sorted(os.listdir(path))
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')

    # Remove any other file that is not a folder with a numer as name
    subjects = [subject for subject in subjects if subject.isdigit()]

    # Randomly choose the train subjects and test subjects or use the given list of train subjects
    if not test_data_only:
        print("train_test_index_list = ", train_test_index_list)
        if train_test_index_list is None:
            train_subjects = random.sample(subjects, amount_train_subjects)
            test_subjects = [subject for subject in subjects if subject not in train_subjects]
        else:    
            train_subjects = [subjects[int(i)-1] for i in train_test_index_list]
            test_subjects = [subject for subject in subjects if subject not in train_subjects]
        print('Train subjects LOL: %s' % train_subjects)
        print('Test subjects: %s' % test_subjects)
    else:
        # no train subjects
        train_subjects = []
        test_subjects = subjects
        print('Test subjects: %s' % test_subjects)

    # subsample also validation subjects
    if datasettype == 'UMAMBA':
        val_subjects = random.sample(train_subjects, int((1-train_val_split)*len(train_subjects)))
        train_subjects = [subject for subject in train_subjects if subject not in val_subjects]

        print('BEFORE REMOVING Validation subjects (Subset of train subjects from before): %s' % val_subjects)
        print('Train subjects: %s' % train_subjects)
        
        print('Validation subjects (Subset of train subjects from before): %s' % val_subjects)
        print('Train subjects: %s' % train_subjects)
            
    # Move the images and labels to the correct folder
    for subject in subjects:
        subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
        if '.DS_Store' in subject_dir_list:
            subject_dir_list.remove('.DS_Store')
        #print(subject_dir_list)
        if not test_data_only:
            image_name = next((s for s in subject_dir_list if modality in s and 'mask' not in s), None)
            mask_name = next((s for s in subject_dir_list if 'mask' in s and modality in s), None)
        else:
            image_name = next((s for s in subject_dir_list if modality in s and 'mask' not in s), None)
        print("Image_name = ", image_name, "and mask_name = ", mask_name)

       # If one of 'T1', 'T2', 'FLAIR', 'T1gd' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
       # If it is not in the image_name, continue with the next subject
        mod_checker = any(mod in image_name for mod in ['T1', 'T2', 'FLAIR', 'T1gd'])
        if modality not in image_name and mod_checker:
            print("Modality = %s" % modality, "Image_name = %s" % image_name)
            raise ValueError('Modality of the image does not match the modality argument')

        # Move the images to the correct folder
        #print('Moving image %s to %s' % (image_name, os.path.join(path, folder_name, 'images_Tr', subject + '_image.nii.gz')))
        if subject in train_subjects:
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Tr', subject + '_image.nii.gz'))
            path_to_save_img_train = os.path.join(new_dataset_path, train_folder_name, subject + '_' + add_id_img + identifier + args.fileending)
            img = nib.load(os.path.join(path, subject, image_name))
            nib.save(img, path_to_save_img_train)
        
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Tr', subject + '_image.nii'))
        
            # Move the labels to the correct folder
            lab = nib.load(os.path.join(path, subject, mask_name))
            path_to_save_lab = os.path.join(new_dataset_path, label_folder_name, subject + add_id_lab + args.fileending)
            nib.save(lab, path_to_save_lab)
        elif datasettype == 'UMAMBA' and subject in val_subjects:
            # Move the images to the correct folder
            path_to_save_img_val = os.path.join(new_dataset_path, val_folder_name, subject + '_' + add_id_img + identifier + args.fileending)
            img = nib.load(os.path.join(path, subject, image_name))
            nib.save(img, path_to_save_img_val)
        
            # Move the labels to the correct folder
            lab = nib.load(os.path.join(path, subject, mask_name))
            path_to_save_lab = os.path.join(new_dataset_path, label_val_folder_name, subject + add_id_lab + args.fileending)
            nib.save(lab, path_to_save_lab)
        else:
            # Move the images to the correct folder (just for test subjects)
            path_to_save_img_test = os.path.join(new_dataset_path, test_folder_name, subject + '_' + add_id_img + identifier + args.fileending)
            img = nib.load(os.path.join(path, subject, image_name))
            nib.save(img, path_to_save_img_test)
    



if __name__ == '__main__':
    print('Starting reorganizing Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/Users/user/Documents/projects_/ASCHOPLEX/ANON_DATA_01_labels')
    parser.add_argument('--task_id', type=int, default=53)
    parser.add_argument('--task_name', type=str, default='Choroid_Plexus')
    parser.add_argument('--test_data_only', type=bool, default=False, help='If True, only the test data is created and masks do not exist')
    parser.add_argument('--amount_train_subjects', type=int, default=10)
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--train_test_index_list', type=str, default=None)
    parser.add_argument('--datasettype', type=str, default='ASCHOPLEX', choices=['ASCHOPLEX', 'NNUNETV2', 'UMAMBA'])
    parser.add_argument('--modality', type=str, default='T1')
    parser.add_argument('--add_id_img', type=str, default='image')
    parser.add_argument('--add_id_lab', type=str, default='seg')
    parser.add_argument('--fileending', type=str, default='.nii')
    args = parser.parse_args()

    # Convert the string back to a list
    train_test_index_list = args.train_test_index_list.split(',')

    folderstructure_changer(args.path, args.task_id, args.task_name,
                            args.test_data_only, args.amount_train_subjects,
                            args.train_val_split, train_test_index_list,
                            args.datasettype, args.modality, args.add_id_img, args.add_id_lab)
    print('Reorganizing Dataset finished')
    

    '''
    python3 launching_tool.py --dataroot '/Users/user/Documents/projects_/ASCHOPLEX/dataset' --work_dir '/Users/user/Documents/projects_/ASCHOPLEX' --finetune yes --prediction yes

    Example Usage:
    python3 step1_dataset_creator.py --path /home/linuxuser/user/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_AP' --train_test_index_list "093,040,032,096,053,017,044,011,054,009,072,008,067,003,092,002,076,068,029,037,018,041,100,004,036,090,043,071,061,038,103,077,022,013,101,094,066,060,079,001,033,058,021,030,056,069,063,015,097,059,057,046,012,099,089,048,024,098,075,042,078,023,087,034,028,039,050,027,025,055,052,014,049,081,085,010" --datasettype 'ASCHOPLEX' --modality 'T1' 
    
    
    /Users/user/Documents/projects_/ASCHOPLEX/launching_tool.py
    --path /Users/user/Documents/projects_/ASCHOPLEX/ANON_DATA
    '''
