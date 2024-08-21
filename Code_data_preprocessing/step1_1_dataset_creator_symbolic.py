import os
from typing import Union, List
import argparse
import random
from random import sample
import shutil
import nibabel as nib
from typing import Union, List
import json

central_data_dir = "/home/linuxlia/Lia_Masterthesis/data/pazienti"


def check_modality(image_name, modality):
    '''
    Check whether the modality of the image matches the modality argument
    
    :param image_name: name of the image
    :param modality: modality of the image
    :return: None
    '''
    mod_checker = any(mod in image_name for mod in ['T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR'])
    if modality not in image_name and mod_checker:
        raise ValueError('Modality of the image does not match the modality argument')
    
def get_image_and_mask_name(subject_dir_list, modality):
    '''
    Get the image and mask name from the list of the subject directory
    
    :param subject_dir_list: list of the subject directory
    :param modality: modality of the image
    :return: image_name, mask_name
    '''
    image_name = next((s for s in subject_dir_list if modality in s and 'mask' not in s), None)
    mask_name = next((s for s in subject_dir_list if 'mask' in s and modality in s), None)
    return image_name, mask_name

def folderstructure_changer_symbolic(path, 
                            task_id = 53,
                            task_name = "Choroid_Plexus",
                            test_data_only = False, 
                            amount_train_subjects:int =5, 
                            train_val_split:float = 0.8,
                            train_test_index_list: Union[List, None] = None, 
                            datasettype:str ='ASCHOPLEX', 
                            modality:str ='T1', 
                            add_id_img:str = 'image', 
                            add_id_lab:str = 'seg', 
                            fileending:str = '.nii',
                            use_single_label_for_bichannel:bool = False, 
                            umamba_fold_json_path:str = None, 
                            skip_validation:bool = False, 
                            output_dir:str = None):
    """
    This function changes the folder structure of the datasets
    from the following structure:
    DATASET
        -MRI_IDsj
            -T1.nii / FLAIR.nii / T1xFLAIR.nii / T2.nii / T1gd.nii / image.nii
            -mask.nii / seg.nii
        -MRI_IDsj
            -T1.nii / FLAIR.nii / T1xFLAIR.nii / T2.nii / T1gd.nii / image.nii
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
    |-imagesVal (optional)
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
    |-labelsVal (optional)
        |-MRI_IDsj.nii.gz
        |- ...
    
    where images_Tr/ imagesTr is the folder containing the training images, 
     imagesVal is the folder containing the validation images
       and images_Ts /imagesTs is the folder containing the test images 
    from which amount_test_subjects are randomly chosen from the original dataset.
    In case of datasettype = NNUNETV2, the images are named MRI_IDsj_{XXXX}.nii and the labels are named MRI_IDsj.nii, where XXXX is the 4-digit modality/channel identifier 
    - FLAIR (0000), T1 (0001), T1gd (0002), T2 (0003), T1xFLAIR (0004).
    - One id can have multiple modalities, e.g. MRI_IDsj_{XXXX}.nii, MRI_IDsj_{XXXY}.nii, MRI_IDsj_{XXYZ}.nii,...}.nii

    :param path: path to the dataset
    :param task_id: id of the task
    :param task_name: name of the task
    :param test_data_only: if True, only the test data is created and masks do not exist
    :param amount_train_subjects: amount of subjects to be used for training 
    :param datasettype: type of the dataset, either ASCHOPLEX or NNUNETV2 or UMAMBA
    :param modality: modality of the images, either T1, T2, FLAIR, T1gd, T1xFLAIR. Can be a list of modalities.
    :param add_id_img: additional identifier for the image files, default is 'image'. Could be ''. 
    :param add_id_lab: additional identifier for the label files, default is 'seg'. Could be ''.
    :param use_single_label_for_bichannel: if True, only one label (T1xFLAIR) is used for the bichannel images, otherwise the corresponding labels for each channel are used
    :param umamba_fold_json_path: path to the json file containing the folds for the UMAMBA dataset
    :param fileending: file ending of the images, default is '.nii'
    :param train_test_index_list: list of indices for the train and test subjects
    :param skip_validation: if True, the validation data is skipped
    :param output_dir: path to the output directory

    :return:
    
    """

    
    # just created simlink to reference labels which are the {}_ChP_mask_T1xFLAIR_manual_seg.nii files in each patient folder

    #------------------------------
    # REFERENCE DATASET
    #------------------------------
    if datasettype == 'reference':
        folder_name = "reference_labels"

        if output_dir is not None:
            new_dataset_path = os.path.join(output_dir, folder_name)
        else:
            new_dataset_path = os.path.join(path, '..', folder_name)

        if train_test_index_list:
            # If indices are provided, seperate into training and testing
            train_ref_dir =  os.path.join(new_dataset_path, "ref_labelTr")
            test_ref_dir =  os.path.join(new_dataset_path, "ref_labelTs")
            os.makedirs(train_ref_dir, exist_ok=True)
            os.makedirs(test_ref_dir, exist_ok=True)

            # Create pattern for the reference label files
            file_pattern = '{}_ChP_mask_T1xFLAIR_manual_seg.nii'

            print("train_test_index_list = ", train_test_index_list)
            print("train_dir = ", train_ref_dir)
            print("test_dir = ", test_ref_dir)
        else:
            # If indices are not provided, throw an error
            raise ValueError('Indices for train and test subjects are not provided')

    else:

        # Create the folder structure
        folder_name = "Dataset%03.0d_%s" % (task_id, task_name)

        if output_dir is not None:
            new_dataset_path = os.path.join(output_dir, folder_name)
        else:
            new_dataset_path = os.path.join(path, '..', folder_name)

        if umamba_fold_json_path is None:
            os.makedirs(new_dataset_path, exist_ok=True)

        if datasettype == 'ASCHOPLEX':
            train_folder_name = 'image_Tr'
            test_folder_name = 'image_Ts'
            label_folder_name = 'label_Tr'
        elif datasettype == 'NNUNETV2':
            train_folder_name = 'imagesTr'
            test_folder_name = 'imagesTs'
            label_folder_name = 'labelsTr'
            add_id_img = ''
            add_id_lab = ''
        elif datasettype == 'UMAMBA':
            train_folder_name = 'imagesTr'
            val_folder_name = 'imagesVal'
            test_folder_name = 'imagesTs'
            label_folder_name = 'labelsTr'
            label_val_folder_name = 'labelsVal'
            add_id_img = ''
            add_id_lab = ''

        if modality == 'FLAIR':
            identifier = '0000'
        elif modality == 'T1':
            identifier = '0001'
        elif modality == 'T1gd':
            identifier = '0002'
        elif modality == 'T2':
            identifier = '0003'
        elif modality == 'T1xFLAIR':
            identifier = '0004'
        else:
            raise ValueError('Modality not supported')
        
        # Label identifier: Only use 0004 for the label if the single label for the bichannel images is used
        identifier_lab = '0004' if use_single_label_for_bichannel else identifier

        # Folder creation
        if not test_data_only and umamba_fold_json_path is None:
            os.makedirs(os.path.join(new_dataset_path, train_folder_name), exist_ok=True)
            os.makedirs(os.path.join(new_dataset_path, label_folder_name), exist_ok=True)

        if datasettype == 'UMAMBA':
            if umamba_fold_json_path is not None:
                print("UMAMBA dataset with fold json provided - skipping directory creation for now.")
            elif not skip_validation:   
                os.makedirs(os.path.join(new_dataset_path, val_folder_name), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, label_val_folder_name), exist_ok=True)
            else:
                os.makedirs(os.path.join(new_dataset_path, test_folder_name), exist_ok=True)
        else: 
            os.makedirs(os.path.join(new_dataset_path, test_folder_name), exist_ok=True)
            print('Folder structure created at %s' % new_dataset_path)

   

    #--------------- Creating lists to later generate the symbolic links -------------------
        
    # Get the list of all the subjects
    subjects = sorted(os.listdir(path))
    if '.DS_Store' in subjects:
        subjects.remove('.DS_Store')

    # Remove any other file that is not a folder with a number as name
    subjects = [subject for subject in subjects if subject.isdigit()]

    # Randomly choose the train subjects and test subjects or use the given list of train subjects
    if test_data_only:
        # no train subjects
        train_subjects = []
        test_subjects = subjects
        print('Test subjects: %s' % test_subjects)

    elif train_test_index_list is not None:
        print("train_test_index_list = ", train_test_index_list)
        train_subjects = [subjects[int(i)-1] for i in train_test_index_list]
        test_subjects = [subject for subject in subjects if subject not in train_subjects]

    else:  # No further information, sample randomly
        print("--- info: Samling randomly")
        train_subjects = random.sample(subjects, amount_train_subjects)
        test_subjects = [subject for subject in subjects if subject not in train_subjects] 

    if umamba_fold_json_path is not None:
        with open(umamba_fold_json_path, 'r') as f:
            folds = json.load(f)    
        # Examinge which subjects in general are used in the folds
        train_subjects = []
        for fold_name, indices in folds.items():
            train_subjects.extend(indices)
        test_subjects = [subject for subject in subjects if subject not in train_subjects]

        # Create train_subject_list and val_subject_list of length num_fold
        # containing each a list of subjects for each fold
        train_subject_all_folds = []
        val_subject_all_folds = []
        for fold_name, val_indices in folds.items():
            print(f"Fold {fold_name} with validation indices {val_indices}")
            train_subjects_fold = [idx for idx in train_subjects if idx not in val_indices]
            val_subjects_fold = val_indices
            train_subject_all_folds.append(train_subjects_fold)
            val_subject_all_folds.append(val_subjects_fold)
        
    
        
    # ------------------ Reading the subject lists, create paths and symbolic links ------------------

    # ------------------
    #  REFERENCE DATASET
    # ------------------

    # Create symbolic links to the images and masks
    if datasettype == 'reference':
        
        if train_test_index_list is not None:
            for subject in train_subjects:
                src_lab_path = os.path.join(path, subject, file_pattern.format(subject)) # source label path
                path_to_save_lab = os.path.join(new_dataset_path, train_ref_dir, file_pattern.format(subject))
                os.symlink(src_lab_path, path_to_save_lab)
                print("Train: created symbolic link from %s to %s" % (src_lab_path, path_to_save_lab)) 

            for subject in test_subjects:
                src_lab_path = os.path.join(path, subject, file_pattern.format(subject))
                path_to_save_lab = os.path.join(new_dataset_path, test_ref_dir, file_pattern.format(subject))
                os.symlink(src_lab_path, path_to_save_lab)
                print("Test: created symbolic link from %s to %s" % (src_lab_path, path_to_save_lab)) 

        else:
            raise ValueError('Indices for train and test subjects are not provided')

    
    # ------------------
    #  UMAMBA DATASET 
    # ------------------

    # randomly subsample also validation subjects or use the given list of folds for train and validation # TODO: include and generate the lists
    elif datasettype == 'UMAMBA':

        if umamba_fold_json_path is not None:
        
            for idx, (train_subjects_fold, val_subjects_fold) in enumerate(zip(train_subject_all_folds, val_subject_all_folds)):
                folder_name = f"Dataset{task_id:03.0f}_{task_name}_fold{idx}"
                new_dataset_path = os.path.join(path, '..', folder_name)
                os.makedirs(new_dataset_path, exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, train_folder_name), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, val_folder_name), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, label_folder_name), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, label_val_folder_name), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_path, test_folder_name), exist_ok=True)

                for subject in train_subjects_fold: # TODO: continue here!!!!!
                    # Create list containing all files for that subject
                    subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                    if '.DS_Store' in subject_dir_list:
                        subject_dir_list.remove('.DS_Store')
                    image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                    # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                    # If it is not in the image_name, continue with the next subject
                    check_modality(image_name, modality)
                    
                    # Create the symbolic links
                    path_to_save_img_train = os.path.join(new_dataset_path, train_folder_name, subject + '_' + add_id_img + identifier + fileending)
                    src_img_path = os.path.join(path, subject, image_name)
                    os.symlink(src_img_path, path_to_save_img_train)
                    print("Train: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_train))
                
                    src_lab_path = os.path.join(path, subject, mask_name)
                    path_to_save_lab = os.path.join(new_dataset_path, label_folder_name, subject + add_id_lab + fileending)
                    # skip if symlink already exists
                    if not os.path.exists(path_to_save_lab):
                        os.symlink(src_lab_path, path_to_save_lab)

                for subject in val_subjects_fold:
                    # Create list containing all files for that subject
                    subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                    if '.DS_Store' in subject_dir_list:
                        subject_dir_list.remove('.DS_Store')
                    image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                    # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                    # If it is not in the image_name, continue with the next subject
                    check_modality(image_name, modality)

                    # Create the symbolic links
                    path_to_save_img_val = os.path.join(new_dataset_path, val_folder_name, subject + '_' + add_id_img + identifier + fileending)
                    src_img_path = os.path.join(path, subject, image_name)
                    os.symlink(src_img_path, path_to_save_img_val)
                    print("Validation: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_val))
                
                    src_lab_path = os.path.join(path, subject, mask_name)
                    path_to_save_lab = os.path.join(new_dataset_path, label_val_folder_name, subject + add_id_lab + fileending)
                    if not os.path.exists(path_to_save_lab):
                        os.symlink(src_lab_path, path_to_save_lab)

                for subject in test_subjects:
                    # Create list containing all files for that subject
                    subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                    if '.DS_Store' in subject_dir_list:
                        subject_dir_list.remove('.DS_Store')
                    image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)

                    # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                    # If it is not in the image_name, continue with the next subject
                    check_modality(image_name, modality)

                    # Create the symbolic links
                    path_to_save_img_test = os.path.join(new_dataset_path, test_folder_name, subject + '_' + add_id_img + identifier + fileending)
                    src_img_path = os.path.join(path, subject, image_name)
                    os.symlink(src_img_path, path_to_save_img_test)
                    print("Test: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_test))

        elif skip_validation:
            for subject in train_subjects:
                # Create list containing all files for that subject
                subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                if '.DS_Store' in subject_dir_list:
                    subject_dir_list.remove('.DS_Store')
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                # If it is not in the image_name, continue with the next subject
                check_modality(image_name, modality)
                    
                # Create the symbolic links
                path_to_save_img_train = os.path.join(new_dataset_path, train_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_train)
                print("Train: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_train))
                
                src_lab_path = os.path.join(path, subject, mask_name)
                path_to_save_lab = os.path.join(new_dataset_path, label_folder_name, subject + add_id_lab + fileending)
                # skip if symlink already exists
                if not os.path.exists(path_to_save_lab):
                    os.symlink(src_lab_path, path_to_save_lab)
                    print("Train: created symbolic link from %s to %s" % (src_lab_path, path_to_save_lab))

            for subject in test_subjects:
                # Create list containing all files for that subject
                subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                if '.DS_Store' in subject_dir_list:
                    subject_dir_list.remove('.DS_Store')
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                # If it is not in the image_name, continue with the next subject
                check_modality(image_name, modality)

                # Create the symbolic links
                path_to_save_img_test = os.path.join(new_dataset_path, test_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_test)
                print("Test: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_test))

        else: # subjsample val_subjects from train_subjects
            val_subjects = random.sample(train_subjects, int((1-train_val_split)*len(train_subjects)))
            train_subjects = [subject for subject in train_subjects if subject not in val_subjects]

            for subject in train_subjects:
                # Create list containing all files for that subject
                subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                if '.DS_Store' in subject_dir_list:
                    subject_dir_list.remove('.DS_Store')
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                # If it is not in the image_name, continue with the next subject
                check_modality(image_name, modality)
                    
                # Create the symbolic links
                path_to_save_img_train = os.path.join(new_dataset_path, train_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_train)
                print("Train: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_train))
                
                src_lab_path = os.path.join(path, subject, mask_name)
                path_to_save_lab = os.path.join(new_dataset_path, label_folder_name, subject + add_id_lab + identifier_lab + fileending)
                # skip if symlink already exists
                if not os.path.exists(path_to_save_lab):
                    os.symlink(src_lab_path, path_to_save_lab)

            for subject in val_subjects:
                # Create list containing all files for that subject
                subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                if '.DS_Store' in subject_dir_list:
                    subject_dir_list.remove('.DS_Store')
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
                    
                # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                # If it is not in the image_name, continue with the next subject
                check_modality(image_name, modality)

                # Create the symbolic links
                path_to_save_img_val = os.path.join(new_dataset_path, val_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_val)
                print("Validation: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_val))
                
                src_lab_path = os.path.join(path, subject, mask_name)
                path_to_save_lab = os.path.join(new_dataset_path, label_val_folder_name, subject + add_id_lab + identifier_lab + fileending)
                if not os.path.exists(path_to_save_lab):
                    os.symlink(src_lab_path, path_to_save_lab)

            for subject in test_subjects:
                # Create list containing all files for that subject
                subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
                if '.DS_Store' in subject_dir_list:
                    subject_dir_list.remove('.DS_Store')
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)

                # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
                # If it is not in the image_name, continue with the next subject
                check_modality(image_name, modality)

                # Create the symbolic links
                path_to_save_img_test = os.path.join(new_dataset_path, test_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_test)
                print("Test: created symbolic link from %s to %s" % (src_img_path, path_to_save_img_test))

    else:
        # Get the list of all the subjects to get the source paths
        for subject in subjects:
            subject_dir_list = sorted(os.listdir(os.path.join(path, subject)), key=str.lower)
            if '.DS_Store' in subject_dir_list:
                subject_dir_list.remove('.DS_Store')
           
            if not test_data_only:
                image_name, mask_name = get_image_and_mask_name(subject_dir_list, modality)
            else:
                image_name = next((s for s in subject_dir_list if modality in s and 'mask' not in s), None)
            print("Image_name = ", image_name, "and mask_name = ", mask_name)

            # If one of 'T1', 'T2', 'FLAIR', 'T1gd', 'T1xFLAIR' is in the image_name, check whether it is the same as the argument modality and throw an error if they differ. 
            # If it is not in the image_name, continue with the next subject
            check_modality(image_name, modality)
            
            
            # Create the symbolic links
            if subject in train_subjects:
                path_to_save_img_train = os.path.join(new_dataset_path, train_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_train)
            
                src_lab_path = os.path.join(path, subject, mask_name)
                path_to_save_lab = os.path.join(new_dataset_path, label_folder_name, subject + add_id_lab + identifier_lab + fileending)
                # skip if symlink already exists
                if not os.path.exists(path_to_save_lab):
                    os.symlink(src_lab_path, path_to_save_lab)

            elif datasettype == 'UMAMBA' and subject in val_subjects:
                path_to_save_img_val = os.path.join(new_dataset_path, val_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_val)
            
                src_lab_path = os.path.join(path, subject, mask_name)
                path_to_save_lab = os.path.join(new_dataset_path, label_val_folder_name, subject + add_id_lab + identifier_lab + fileending)
                if not os.path.exists(path_to_save_lab):
                    os.symlink(src_lab_path, path_to_save_lab)
    
            else:
                # (just for test subjects)
                path_to_save_img_test = os.path.join(new_dataset_path, test_folder_name, subject + '_' + add_id_img + identifier + fileending)
                src_img_path = os.path.join(path, subject, image_name)
                os.symlink(src_img_path, path_to_save_img_test)
        
                


if __name__ == '__main__':
    print('Starting reorganizing Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/linuxlia/Lia_Masterthesis/data/pazienti', help='Path to the dataset')
    parser.add_argument('--task_id', type=int, default=53)
    parser.add_argument('--task_name', type=str, default='Choroid_Plexus')
    parser.add_argument('--test_data_only', type=bool, default=False, help='If True, only the test data is created and masks do not exist')
    parser.add_argument('--amount_train_subjects', type=int, default=10)
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--train_test_index_list', type=str, default=None)
    parser.add_argument('--datasettype', type=str, default='ASCHOPLEX', choices=['ASCHOPLEX', 'NNUNETV2', 'UMAMBA', 'reference'])
    parser.add_argument('--modality', type=str, nargs='+', default=['T1'])
    parser.add_argument('--add_id_img', type=str, default='image')
    parser.add_argument('--add_id_lab', type=str, default='seg')
    parser.add_argument('--fileending', type=str, default='.nii')
    parser.add_argument('--use_single_label_for_bichannel', type=bool, default=False, help='If True, only one label (T1xFLAIR) is used for the bichannel images, otherwise the corresponding labels for each channel are used')
    parser.add_argument('--umamba_fold_json_path', type=str, default=None, help='Path to the json file containing the folds for the UMAMBA dataset')
    parser.add_argument('--skip_validation', type=bool, default=False, help='If True, the validation data is not created')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory')
    args = parser.parse_args()

    print(f"args = {args}")

    # Convert the string back to a list
    if args.train_test_index_list is not None:
        args.train_test_index_list = args.train_test_index_list.split(',')

    # if modality is not a list, convert it to a list of one element, else, ierate over the list and create the symbolic links for each modality
    if not isinstance(args.modality, list):
        modality = [args.modality]
    else:
        print("NO")
    for mod in args.modality:
        print(f"modality = {mod}, type = {type(mod)}")
        folderstructure_changer_symbolic(args.path, args.task_id, args.task_name,
                            args.test_data_only, args.amount_train_subjects,
                            args.train_val_split, args.train_test_index_list,
                            args.datasettype, mod, args.add_id_img, args.add_id_lab, 
                            args.fileending, args.use_single_label_for_bichannel, 
                            args.umamba_fold_json_path, args.skip_validation, args.output_dir)


  
    print('Reorganizing Dataset finished')
    

    '''
    python3 launching_tool.py --dataroot '/Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX/dataset' --work_dir '/Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX' --finetune yes --prediction yes

    Example Usage:
    python step1_1_dataset_creator_symbolic.py --path /var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels --folder_name dataset_aschoplex --amount_train_subjects 10
    python step1_1_dataset_creator_symbolic.py --path /var/data/MONAI_Choroid_Plexus/ANON_FLAIR_COREG_2 --folder_name dataset_aschoplex_2 --test_data_only True
    python3 step1_1_dataset_creator_symbolic.py --path /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/phuse_thesis_2024/data/ANON_FLAIR_COREG_2 --folder_name dataset_aschoplex_2 --test_data_only True
    python3 step1_1_dataset_creator_symbolic.py --path /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/phuse_thesis_2024/data/ANON_FLAIR_COREG_2 --folder_name dataset_aschoplex_2 --test_data_only True
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/ANON_FLAIR_COREG --task_id 59 --task_name 'ChoroidPlexus_FLAIR' --datasettype 'NNUNETV2' --modality 'FLAIR' 
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/ANON_FLAIR_COREG --task_id 59 --task_name 'ChoroidPlexus_FLAIR' --datasettype 'UMAMBA' --modality 'FLAIR' --add_id_img '' --add_id_lab '' --fileending '.nii.gz'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 59 --task_name 'ChoroidPlexus_FLAIR' --datasettype 'UMAMBA' --modality 'FLAIR' --add_id_img '' --add_id_lab '' --fileending '.nii.gz'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 12 --task_name 'ChoroidPlexus_T1' --train_test_index_list "001,004,006,014,027,101" --datasettype 'UMAMBA' --modality 'FLAIR' --add_id_img '' --add_id_lab '' --fileending '.nii.gz'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 9 --task_name 'ChoroidPlexus_T1_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 9 --task_name 'ChoroidPlexus_T1_sym_UMAMBA' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'UMAMBA' --modality 'T1' 
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 9 --task_name 'ChoroidPlexus_FLAIR_sym_UMAMBA' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'UMAMBA' --modality 'FLAIR' 
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 8 --task_name 'ChoroidPlexus_FLAIR_sym_REFERENCETEST' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'reference' --modality 'FLAIR' 
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'reference' 
    
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_FLAIR_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1xFLAIR_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1xFLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_FLAIR_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 'FLAIR' --use_single_label_for_bichannel True
    
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 22 --task_name 'ChoroidPlexus_T1_sym_PHU' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 22 --task_name 'ChoroidPlexus_FLAIR_sym_PHU' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 22 --task_name 'ChoroidPlexus_T1xFLAIR_sym_PHU' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1xFLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 22 --task_name 'ChoroidPlexus_T1_FLAIR_sym_PHU' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 22 --task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_PHU' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 'FLAIR' --use_single_label_for_bichannel True
    

    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_sym_UMAMBA' --umamba_fold_json_path '/home/linuxlia/Lia_Masterthesis/data/pazienti/folds.json' --datasettype 'UMAMBA' --add_id_img '' --add_id_lab '' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_sym_UMAMBA'  --datasettype 'UMAMBA' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --add_id_img '' --add_id_lab '' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_FLAIR_sym_UMAMBA' --umamba_fold_json_path '/home/linuxlia/Lia_Masterthesis/data/pazienti/folds.json' --datasettype 'UMAMBA' --add_id_img '' --add_id_lab '' --modality 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1xFLAIR_sym_UMAMBA' --umamba_fold_json_path '/home/linuxlia/Lia_Masterthesis/data/pazienti/folds.json' --datasettype 'UMAMBA' --add_id_img '' --add_id_lab '' --modality 'T1xFLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_FLAIR_sym_UMAMBA' --umamba_fold_json_path '/home/linuxlia/Lia_Masterthesis/data/pazienti/folds.json' --datasettype 'UMAMBA' --add_id_img '' --add_id_lab '' --modality 'T1' 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA' --umamba_fold_json_path '/home/linuxlia/Lia_Masterthesis/data/pazienti/folds.json' --datasettype 'UMAMBA' --add_id_img '' --add_id_lab '' --modality 'T1' 'FLAIR' --use_single_label_for_bichannel True
    
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_sym_UMAMBA'  --datasettype 'UMAMBA' --output_dir "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/" --skip_validation True --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --add_id_img '' --add_id_lab '' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_FLAIR_sym_UMAMBA' --datasettype 'UMAMBA' --output_dir "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/" --skip_validation True --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --skip_validation True --modality 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1xFLAIR_sym_UMAMBA' --datasettype 'UMAMBA' --output_dir "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/" --skip_validation True --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --modality 'T1xFLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_FLAIR_sym_UMAMBA' --datasettype 'UMAMBA' --output_dir "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/" --skip_validation True --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --modality 'T1' 'FLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 33 --task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA' --datasettype 'UMAMBA' --output_dir "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/" --skip_validation True --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --modality 'T1' 'FLAIR' --use_single_label_for_bichannel True
    
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 3 --task_name 'ChoroidPlexus_T1_sym_UMAMBA' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'UMAMBA' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 390 --task_name 'ChoroidPlexus_T1xFLAIR_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1xFLAIR'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 391 --task_name 'ChoroidPlexus_T1andFLAIR_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality T1 FLAIR
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 111 --task_name 'ChoroidPlexus_T1_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1'
    python3 step1_1_dataset_creator_symbolic.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 111 --task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_AP' --train_test_index_list "056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" --datasettype 'ASCHOPLEX' --modality 'T1' 'FLAIR' --use_single_label_for_bichannel True
                                                                                                                     
    
    /Users/liaschmid/Documents/Uni Heidelberg/7_Semester_Thesis/ASCHOPLEX/launching_tool.py
    --path /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/ASCHOPLEX/ANON_DATA
    '''


