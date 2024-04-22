import argparse
import os
import random
import shutil
import nibabel as nib

def folderstructure_changer(path, folder_name, test_data_only = False, amount_train_subjects=5):
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
    where MRI_IDsj is the subject ID from 00 to N 
    to the following structure:
    
    DATASET
    |-image_Tr
        |-MRI_IDsj_image.nii.gz
        |- ....
    |-image_Ts
        |-MRI_IDsj_image.nii.gz
        |- ....
    |-label_Tr
        |-MRI_IDsj_seg.nii.gz
        |- ....
    
    where images_Tr is the folder containing the training images and images_Ts is the folder containing the test images 
    from which amount_test_subjects are randomly chosen from the original dataset.
    :param path: path to the dataset
    :param folder_name: name of the folder to be created
    :param test_data_only: if True, only the test data is created and masks do not exist
    :param amount_train_subjects: amount of subjects to be used for training 
    :return:

    
    """


    # Create the folder structure
    new_dataset_path = os.path.join(path, '..', folder_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    if not test_data_only:
        os.makedirs(os.path.join(new_dataset_path, 'image_Tr'), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, 'label_Tr'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'image_Ts'), exist_ok=True)
    print('Folder structure created')
    

    # Get the list of all the subjects
    subjects = os.listdir(path)
    #subjects.remove(folder_name)
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')

    # Randomly choose the train subjects and test subjects
    if not test_data_only:
        train_subjects = random.sample(subjects, amount_train_subjects)
        test_subjects = [subject for subject in subjects if subject not in train_subjects]
        print('Train subjects: %s' % train_subjects)
        print('Test subjects: %s' % test_subjects)
    else:
        train_subjects = []
        test_subjects = subjects
        print('Test subjects: %s' % test_subjects)
            
    # Move the images and labels to the correct folder
    for subject in subjects:
        subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
        if '.DS_Store' in subject_dir_list:
            subject_dir_list.remove('.DS_Store')
        print(subject_dir_list)
        if not test_data_only:
            image_name = subject_dir_list[1] if 'mask' in subject_dir_list[0] else subject_dir_list[0]
            mask_name = subject_dir_list[0] if 'mask' in subject_dir_list[0] else subject_dir_list[1]
        else:
            image_name = subject_dir_list[0]
        
        #image_name = subject_dir_list[1]
        #mask_name = subject_dir_list[0]

        # Move the images to the correct folder
        #print('Moving image %s to %s' % (image_name, os.path.join(path, folder_name, 'images_Tr', subject + '_image.nii.gz')))
        if subject in train_subjects:
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Tr', subject + '_image.nii.gz'))
            img = nib.load(os.path.join(path, subject, image_name))
            nib.save(img, os.path.join(new_dataset_path, 'image_Tr', subject + '_image.nii'))
        
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Tr', subject + '_image.nii'))
        
            # Move the labels to the correct folder
            lab = nib.load(os.path.join(path, subject, mask_name))
            nib.save(lab, os.path.join(new_dataset_path, 'label_Tr', subject + '_seg.nii'))
            #shutil.copy(os.path.join(path, subject, mask_name), os.path.join(new_dataset_path, 'label_Tr', subject + '_seg.nii.gz'))
            #shutil.copy(os.path.join(path, subject, mask_name), os.path.join(new_dataset_path, 'label_Tr', subject + '_seg.nii'))
        else:
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Ts', subject + '_image.nii.gz'))
            #shutil.copy(os.path.join(path, subject, image_name), os.path.join(new_dataset_path, 'image_Ts', subject + '_image.nii'))

            img = nib.load(os.path.join(path, subject, image_name))
            nib.save(img, os.path.join(new_dataset_path, 'image_Ts', subject + '_image.nii'))
    
    
    # Remove the original folders
    #for subject in subjects:
    #    shutil.rmtree(os.path.join(path, subject))
    # Remove the dataset folder
    #shutil.rmtree(path)
            
    


if __name__ == '__main__':
    print('Starting reorganizing Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/Users/liaschmid/Documents/Uni Heidelberg/7_Semester_Thesis/ASCHOPLEX/ANON_DATA_01_labels')
    parser.add_argument('--folder_name', type=str, default='dataset')
    parser.add_argument('--test_data_only', type=bool, default=False, help='If True, only the test data is created and masks do not exist')
    parser.add_argument('--amount_train_subjects', type=int, default=10)
    args = parser.parse_args()

    folderstructure_changer(args.path, args.folder_name, args.test_data_only, args.amount_train_subjects)
    print('Reorganizing Dataset finished')
    

    '''
    python3 launching_tool.py --dataroot '/Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX/dataset' --work_dir '/Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX' --finetune yes --prediction yes

    Example Usage:
    python step1_dataset_creator.py --path /var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels --folder_name dataset_aschoplex --amount_train_subjects 10
    python step1_dataset_creator.py --path /var/data/MONAI_Choroid_Plexus/ANON_FLAIR_COREG_2 --folder_name dataset_aschoplex_2 --test_data_only True
    python3 step1_dataset_creator.py --path /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/phuse_thesis_2024/data/ANON_FLAIR_COREG_2 --folder_name dataset_aschoplex_2 --test_data_only True
    
    /Users/liaschmid/Documents/Uni Heidelberg/7_Semester_Thesis/ASCHOPLEX/launching_tool.py
    --path /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX/ANON_DATA
    '''
