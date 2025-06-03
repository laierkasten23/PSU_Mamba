import os
import argparse
import json
import random
import math
import ast
from collections import OrderedDict, defaultdict

# https://github.com/FAIR-Unipd/ASCHOPLEX/blob/main/Code/create_json.py
# and https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/notebooks/msd_datalist_generator.ipynb 

def filter_groups(groups, include_groups):
    '''
    Filter the groups dictionary to include only the specified groups.
    
    Args:
    groups (dict): A dictionary mapping image indices to group names.
    include_groups (list): A list of group names to include in the experiment.
    
    Returns:
    dict: A dictionary mapping image indices to group names, filtered to include only the specified groups.
    '''
    print("using include_groups", include_groups)
    print("Original groups", groups)
    filtered_groups = {k: v for k, v in groups.items() if v in include_groups}
    return filtered_groups

def calculate_group_distribution(groups):
    '''
    Calculate the distribution of groups in the dataset.

    Args:
    groups (dict): A dictionary mapping image indices to group names.

    Returns:
    dict: A dictionary mapping group names to the proportion of the dataset they should occupy.
    '''
    group_counts = defaultdict(int)
    for group in groups.values():
        group_counts[group] += 1
    total = sum(group_counts.values())
    group_distribution = {group: count / total for group, count in group_counts.items()}
    return group_distribution

def split_data(groups, group_distribution, split_ratios):
    '''
    Split the data into training, validation, and test sets.

    Args:
    groups (dict): A dictionary mapping image indices to group names.
    group_distribution (dict): A dictionary mapping group names to the proportion of the dataset they should occupy.
    split_ratios (tuple): A tuple of floats representing the proportions of the dataset to allocate to training, validation, and/or test sets.

    Returns:

    tuple: Three lists of image indices representing the training, validation, and/or test sets.
    '''
    num_splits = len(split_ratios)
    if num_splits not in [1, 2, 3]:
        raise ValueError("split_ratios must have 1, 2, or 3 elements.")

    train_ratio = split_ratios[0]
    val_ratio = split_ratios[1] if num_splits > 1 else 0
    test_ratio = split_ratios[2] if num_splits > 2 else 0

    train_indices, val_indices, test_indices = [], [], []
    
    for group, proportion in group_distribution.items():
        group_indices = [int(i)-1 for i, g in groups.items() if g == group]
        random.seed(42)  # Set seed for reproducibility
        random.shuffle(group_indices)  
        
        n_train = int(len(group_indices) * train_ratio)
        n_val = int(len(group_indices) * val_ratio)
        
        train_indices.extend(group_indices[:n_train])
        if num_splits > 1:
            val_indices.extend(group_indices[n_train:n_train + n_val])
        if num_splits > 2:
            test_indices.extend(group_indices[n_train + n_val:])

    result = [train_indices]
    if num_splits > 1:
        result.append(val_indices)
    if num_splits > 2:
        result.append(test_indices)

    return tuple(result)

def split_into_folds(training_indices, groups, num_folds=4, start_one_based_index=True): 
    '''
    Split the training data into N random folds while maintaining balanced group distribution.

    Args:
    training_indices (list of int): List of training indices.
    groups (dict): A dictionary mapping image indices to group names.
    num_folds (int): Number of folds.
    start_one_based_index (bool): Whether the indices start at 1 or 0.

    Returns:
    list: A list of lists, where each sublist represents a fold with indices 
            (starting at 1 default, so be careful with the filenames that start at 0).
    '''
    group_to_indices = defaultdict(list)
    for idx in training_indices:
        # add 1 to idx to match the groups dictionary
        group_to_indices[groups[str(idx+1)]].append(idx)

    folds = [[] for _ in range(num_folds)]  # Initialize empty folds
    for group, indices in group_to_indices.items():
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % num_folds].append(idx)

    if start_one_based_index:
        folds = [[idx + 1 for idx in fold] for fold in folds]

    return folds


def generate_json(args):
    """
    Class for writing .json files to run training/ finetuning/ testing/ training and predicting or finetuning and predicting Choroid Plexus segmentations.

    dataroot = "/var/data/MONAI_Choroid_Plexus/dataset_monai_train_from_scratch"
    work_dir = "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training"
    json_file=WriteTrainJSON(dataroot, work_dir).write_train_val_json(json_filename = "train_val3.json")

    Usage: 
        python step2_create_json_nnunetv2.py --mode "train" --dataroot "/var/data/MONAI_Choroid_Plexus/dataset_monai_train_from_scratch" --work_dir "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training" --train_val_ratio 0.5 --num_folds 5   
        python step2_create_json_nnunetv2.py --mode "test" --dataroot "/var/data/MONAI_Choroid_Plexus/dataset_aschoplex" --work_dir "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training" --train_val_ratio 0.5 --num_folds 5   
        python step2_create_json_nnunetv2.py --mode "train" --dataroot "/home/linuxlia/Lia_Masterthesis/data/dataset_aschoplex" --work_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/monai_segmentation/monai_training/working_directory_0509" --train_val_ratio 0.8 --num_folds 5   
        python step2_create_json_nnunetv2.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset059_ChoroidPlexus_FLAIR" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset059_ChoroidPlexus_FLAIR" --train_val_ratio 0.8 --num_folds 1  
        python step2_create_json_nnunetv2.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset059_ChoroidPlexus_FLAIR" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset059_ChoroidPlexus_FLAIR" --train_val_ratio 0.8 --num_folds 1 --datasettype "NNUNETV2" --modality "['FLAIR']"
        python step2_create_json_nnunetv2.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']"
        python step2_create_json_nnunetv2.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset003_ChoroidPlexus_T1_sym_UMAMBA" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset003_ChoroidPlexus_T1_sym_UMAMBA" --train_val_ratio 1.0 --num_folds 4 --datasettype "UMAMBA" --modality "['T1']"
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']"
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_FLAIR_sym_AP" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' --include_groups "['AD', 'Psy']" --json_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/JSON_file_experiments" (successful :) ) 
 
        
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset002_ChoroidPlexus_T1_sym_PHU" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset002_ChoroidPlexus_FLAIR_sym_PHU" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['FLAIR']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset002_ChoroidPlexus_T1xFLAIR_sym_PHU" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1xFLAIR']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset002_ChoroidPlexus_T1_FLAIR_sym_PHU" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1', 'FLAIR']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset002_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_PHU" --train_val_ratio 1.0 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1', 'FLAIR']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
        

        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP" --train_val_ratio .5 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' --json_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/JSON_file_experiments" (successful :) ) 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP" --train_val_ratio .5 --num_folds 4 --datasettype "ASCHOPLEX" --modality "['T1']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' --json_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/JSON_file_experiments" (successful :) ) 
        python step2b_create_json_nnunetv2_newversion.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset003_ChoroidPlexus_T1_sym_UMAMBA" --train_val_ratio .5 --num_folds 4 --datasettype "UMAMBA" --modality "['T1']" --groups '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' --json_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/JSON_file_experiments" (successful :) ) 
        
        python step2_create_json_nnunetv2.py --mode "test" --dataroot "/var/data/MONAI_Choroid_Plexus/dataset_aschoplex" --work_dir "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training" --train_val_ratio 0.5 --num_folds 5   
        
        

    Args:
        dataroot (str): The root directory of the dataset. Default is ".".
        benchmark_dataroot (str, optional): The root directory of the benchmark dataset. Default is None.
        description (str, optional): Description of the dataset. Default is None.
        mode (str): One of: 'train', 'finetune', 'test', 'train_predict', 'finetune_predict'. Which operation mode is used. 
        work_dir (str): The working directory. Default is args.dataroot.
        json_dir (str): The name of the directory where the json files are stored. Default is None.
        train_val_ratio (float): The ratio of training data to validation data. Default is 0.5.
        num_folds (int): The number of folds to split the training data into. Default is 5.
        indices (list): The indices of the training data. Default is None.
        groups_json_path (str): The path to the groups file. Default is '/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json'.
        include_groups (list): The groups to include in the experiment. Default is None.
        pretrained_model_path (str): The path to the pretrained model. Default is None.
        datasettype (str): The type of dataset. Default is 'ASCHOPLEX'. Other option is 'NNUNETV2' and 'UMAMBA.
        modality (str): The modality of the dataset. Default is 'MR'. List or String. 
        fileending (str): The file ending of the images. Default is '.nii'.
        skip_validation (bool): Skip the validation data. Default is False.

        Writes the .json file based on the provided parameters.
            
        Returns:
            str: The path of the generated .json file.

        --------------------------------
        Folderstructure of dataroot is either: 
        --------------------------------
        i_ChP.nii.gz
        j_ChP.nii.gz
        ...
        labels
            final
                i_ChP.nii.gz
                j_ChP.nii.gz
                ...
        --------------------------------
        or 
        image_Tr
            a_image.nii
            b_image.nii
            ...
        image_Ts
            i_image.nii
            j_image.nii
            ...
        label_Tr
            a_seg.nii
            b_seg.nii
            ...
        --------------------------------
        or 
        imagesTr
            a_image.nii
            b_image.nii
            ...
        imagesTs
            i_image.nii
            j_image.nii
            ...
        labelsTr
            a_seg.nii
            b_seg.nii
            ...

        --------------------------------
        or 
        imagesTr
            a_{XXXX}.nii
            b_{XXXX}.nii
            ...
        (imagesVal)
            c_{XXXX}.nii
            d_{XXXX}.nii
            ...
        imagesTs
            i_{XXXX}.nii
            j_{XXXX}.nii
            ...
        labelsTr
            a.nii
            b.nii
            ...
        (labelsVal)
            c.nii
            d.nii
            ...

        where a, b, c, d, i, j are subject identifiers and {XXXX} the modality.  
        """
    
    # If no working directory is given, save json file in the dataroot directory
    if args.work_dir is None:
        args.work_dir = args.dataroot
        
    # Convert "None" string to None
    if args.groups_json_path == "None":
        args.groups_json_path = None

    # Set path to output file
    if args.json_dir is not None:
        output_folder = os.path.join(args.work_dir, args.json_dir)
    else:
        output_folder = args.work_dir
    
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Set name of the json file
    if args.datasettype == 'NNUNETV2' or args.datasettype == 'UMAMBA':
        json_filename="dataset.json"
    else:
        if args.mode in ['train', 'finetune']:
            json_filename="dataset_train_val.json"
        elif args.mode in ['train_predict', 'finetune_predict']:
            json_filename="dataset_train_val_pred.json"
        else:
            json_filename="dataset_test.json"

    # If UMAMBA, then Training and Validation is already seperated. Assure that train_val_ratio is 1.0
    if args.datasettype == 'UMAMBA':
        args.train_val_ratio = 1.0  

    # create json file - manually set: general information
    #------------------------------------------------------------------------------------------
    json_dict = OrderedDict()
    json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
    json_dict['description'] = args.description if args.description is not None else "Dataset for Choroid Plexus segmentation"
    json_dict['data_benchmark_base_dir'] = args.benchmark_dataroot if args.benchmark_dataroot is not None else None # TODO: Generalize when setting overall projectdirectory for later distribution 
    json_dict['tensorImageSize'] = "3D"
    

    # add channel_names or modality to json file depending on the datasettype (only for NNUNETV2 and UMAMBA and ASCHOPLEX)
    if args.datasettype == 'NNUNETV2':
        json_dict['dataset_type'] = 'NNUNETV2'
    elif args.datasettype == 'UMAMBA':
        json_dict['dataset_type'] = 'UMAMBA'
        
    if args.datasettype == 'NNUNETV2' or args.datasettype == 'UMAMBA':
        # Assume args.modality is a list of modalities e.g. ["T1", "FLAIR"]
        print("args.modality", args.modality)
        channel_names = {str(i): modality for i, modality in enumerate(args.modality)}
        print("channel_names", channel_names)

        json_dict['channel_names'] = channel_names
        json_dict['file_ending'] = args.fileending
        
        json_dict['labels'] = {
        "background": "0",
        "Choroid Plexus": "1"
    }
    else:
        json_dict['dataset_type'] = 'ASCHOPLEX'
        json_dict['modality'] = {
            "0": "MR"
        }
            
        json_dict['labels'] = {
            "0": "background",
            "1": "Choroid Plexus"
        }

    print("json_dict", json_dict)

    # Check whether datarooot is correct
    if not os.path.exists(args.dataroot):
        raise ValueError("The dataroot is not correct. Please, provide the correct path to the dataset.")
    
    #------------------------------------------------------------------------------------------
    # TRAINING and VALIDATION
    # needed when one of the following modes is selected. Testing is handled separately.
    #------------------------------------------------------------------------------------------
    if args.mode in ['train', 'finetune', 'train_predict', 'finetune_predict']:

        print(args.datasettype == 'ASCHOPLEX')
        # Check the folder structure
        if os.path.exists(os.path.join(args.dataroot, 'labels')):
            image_dir = args.dataroot
            label_dir = os.path.join(args.dataroot, 'labels', 'final')
        elif args.datasettype == 'ASCHOPLEX' and os.path.exists(os.path.join(args.dataroot, 'image_Tr')):
            image_dir = os.path.join(args.dataroot, 'image_Tr')
            label_dir = os.path.join(args.dataroot, 'label_Tr')
        elif args.datasettype == 'NNUNETV2' and os.path.exists(os.path.join(args.dataroot, 'imagesTr')):
            image_dir = os.path.join(args.dataroot, 'imagesTr')
            label_dir = os.path.join(args.dataroot, 'labelsTr')
        elif args.datasettype == 'UMAMBA' and os.path.exists(os.path.join(args.dataroot, 'imagesTr')):
            image_dir = os.path.join(args.dataroot, 'imagesTr')
            label_dir = os.path.join(args.dataroot, 'labelsTr')
            if not args.skip_validation:
                val_dir = os.path.join(args.dataroot, 'imagesVal')
                val_label_dir = os.path.join(args.dataroot, 'labelsVal')
        else:
            raise ValueError("The folder structure is not correct. Please, provide the data in the correct format.")
        print("image_dir", image_dir)
        print("label_dir", label_dir)

        filenames_image = os.listdir(image_dir)
        filenames_image.sort()


        # Check if the label directory is in filenames_image and remove it from the list of filenames (needed for MONAI label)
        if 'labels' in filenames_image:
            filenames_image.remove('labels')
        
        filenames_label = os.listdir(label_dir)
        filenames_label.sort()   

        # remove hidden files and .DS_Store files 
        filenames_image = [f for f in filenames_image if not f.startswith('.')]
        filenames_label = [f for f in filenames_label if not f.startswith('.')]
        print("len(filenames_image)", len(filenames_image), "len(filenames_label)", len(filenames_label))
        
        if len(args.modality) == 1 and len(filenames_image)!=len(filenames_label):
            raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")
        if len(args.modality) == 2 and (len(filenames_image)!=2*len(filenames_label) and len(filenames_image)!=len(filenames_label)):    
            raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")

        # List of image and label paths
        image_paths = [os.path.join(image_dir, filename) for filename in sorted(filenames_image)]
        label_paths = [os.path.join(label_dir, filename) for filename in sorted(filenames_label)]
        print("len(image_paths)", len(image_paths), "len(label_paths)", len(label_paths))

       
        # Check that all files have ending .nii or .nii.gz
        for i in range(len(filenames_image)):
            if not(filenames_image[i].endswith('.nii') | filenames_image[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
        for i in range(len(filenames_label)):
            if not(filenames_label[i].endswith('.nii') | filenames_label[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
            
        

        ###

        # If UMAMBA, get validation data
        if args.datasettype == 'UMAMBA':

            if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                json_dict['numTraining'] = int(len(image_paths)/2)

                # Create dictionaries to map base filenames to paths
                image_dict_0000 = {os.path.basename(path).rsplit('_', 1)[0]: path for path in sorted(image_paths) if '0000' in path}
                image_dict_0001 = {os.path.basename(path).rsplit('_', 1)[0]: path for path in sorted(image_paths) if '0001' in path}
                label_dict = {os.path.basename(path).rsplit('_', 1)[0]: path for path in sorted(label_paths)}
       

                # Create the training list
                training_list = []

                for base_filename_tr, base_filename_lab in zip(sorted(image_dict_0000.keys()), sorted(label_dict.keys())):
                    image_path = image_dict_0000[base_filename_tr]
                    image_path2 = image_dict_0001[base_filename_tr]
                    label_path = label_dict[base_filename_lab]
                    training_list.append({
                        "fold": 0,
                        "image": image_path,
                        "image2": image_path2,
                        "label": label_path,
                        "subject": base_filename_tr
                    })
                json_dict['training'] = training_list
            else:
                json_dict['numTraining'] = len(image_paths)
                json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for i, j in zip(image_paths, label_paths)]
            
            if not args.skip_validation:
                # image paths
                filenames_image_val = os.listdir(val_dir)
                filenames_image_val.sort()
                filenames_image_val = [f for f in filenames_image_val if not f.startswith('.')]
                image_paths_val = [os.path.join(val_dir, filename) for filename in filenames_image_val]
                # corresponding label paths
                filenames_label_val = os.listdir(val_label_dir)
                filenames_label_val.sort()
                filenames_label_val = [f for f in filenames_label_val if not f.startswith('.')]
                label_paths_val = [os.path.join(val_label_dir, filename) for filename in filenames_label_val]

                for i in range(len(filenames_image_val)):
                    if not(filenames_image_val[i].endswith('.nii') | filenames_image_val[i].endswith('.nii.gz')):
                        raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                    if not(filenames_label_val[i].endswith('.nii') | filenames_label_val[i].endswith('.nii.gz')):
                        raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                    
                json_dict['numValidation'] = len(image_paths_val)
                json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for i,j in zip(image_paths_val, label_paths_val)]
        
            
        else:   # ASCHOPLEX and NNUNETV2
            ###
            # If groups are provided (not None), filter the groups and calculate the group distribution
            ###
           
            
            if args.groups_json_path:
                with open(args.groups_json_path, 'r') as f:
                    groups_all = json.load(f)
                    # get key index_order_train from json file if in keys else throw error
                    if 'index_order_train' in groups_all.keys():
                        groups = groups_all['index_order_train']
                    else:
                        print("No index_order_train in groups json file. Please, provide the correct json file.")
                        raise ValueError("No index_order_train in groups json file. Please, provide the correct json file.")

                    
                if args.include_groups: # just include the groups that are specified
                    used_groups = filter_groups(groups, args.include_groups)
                    json_dict['included_groups'] = args.include_groups
                else:
                    used_groups = groups    # use all groups

                group_distribution = calculate_group_distribution(used_groups)
                json_dict['group_distribution'] = group_distribution
                print("group_distribution", group_distribution)
                split_ratios = (args.train_val_ratio, 1 - args.train_val_ratio) # TODO: include test data
                indices_tr_val = split_data(used_groups, group_distribution, split_ratios)
                
                # Format indices to be zero-padded to three digits and add 1 to match the filenames (ESSENTIAL)
                formatted_indices_tr_val = [
                    [f"{i+1:03}" for i in sorted(indices_tr_val[0])],
                    [f"{i+1:03}" for i in sorted(indices_tr_val[1])]
                    ]
                print("formatted_indices_tr_val", formatted_indices_tr_val)
                

                # Split data into training and validation based on the formatted indices
                train_ids = []; train_ids2 = [] # for second modality
                for i in sorted(formatted_indices_tr_val[0]):
                    print("i", i)
                    for path in sorted(image_paths):
                        if f"/{i}_" in path:
                            train_ids.append(path)
                            print("train_id added", path)
                            if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                                train_ids2.append(path.replace('0000', '0001'))
                            break
                

                validation_ids = []; validation_ids2 = []
                for i in sorted(formatted_indices_tr_val[1]):
                    for path in sorted(image_paths):
                        if f"{i}_" in path:
                            validation_ids.append(path)
                            if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                                validation_ids2.append(path.replace('0000', '0001'))
                            break

                label_train_ids = []; label_train_ids2 = []
                for i in sorted(formatted_indices_tr_val[0]):
                    for path in sorted(label_paths):
                        if f"{i}seg" in path:
                            label_train_ids.append(path)
                            if '0004' not in path and (args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']):
                                label_train_ids2.append(path.replace('0000', '0001'))
                            break

                label_valid_ids = []; label_valid_ids2 = []
                for i in sorted(formatted_indices_tr_val[1]):
                    for path in sorted(label_paths):
                        if f"{i}seg" in path:
                                label_valid_ids.append(path)
                                if args.modality == ['T1', 'FLAIR'] and '0004' not in path:
                                    label_valid_ids2.append(path.replace('0000', '0001'))
                                break

                json_dict['numTraining'] = len(train_ids)
                json_dict['numValidation'] = len(validation_ids)
                # if modality is T1 and FLAIR, add the second modality as list for 'image'. If there are two labels as well add them as list for 'label', else for one label as before
                if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                    if len(label_train_ids2) != 0 : # two modalities and two labels
                        json_dict['training'] = [{"fold": 0, "image": ['%s' %i, '%s' %j] , "label": ['%s' %k, '%s' %l], "subject_id": '%s' %m} for i, j, k, l, m in zip(train_ids, train_ids2, label_train_ids, label_train_ids2, sorted(formatted_indices_tr_val[0]))]
                        json_dict['validation'] = [{"image": ['%s' %i, '%s' %j], "label": ['%s' %k, '%s' %l]} for i, j, k, l in zip(validation_ids, validation_ids2, label_valid_ids, label_valid_ids2)]
                    else:   # two modalities and one label
                        json_dict['training'] = [{"fold": 0, "image": ['%s' %i, '%s' %j] , "label": ['%s' %k], "subject_id": '%s' %m} for i, j, k, m in zip(train_ids, train_ids2, label_train_ids, sorted(formatted_indices_tr_val[0]))]
                        json_dict['validation'] = [{"image": ['%s' %i, '%s' %j], "label": ['%s' %k]} for i, j, k, l in zip(validation_ids, validation_ids2, label_valid_ids)]
                else:   # only one modality
                    json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j, "subject_id": '%s' %k} for i, j, k in zip(train_ids, label_train_ids, sorted(formatted_indices_tr_val[0]))]
                    json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j, "subject_id": '%s' %k} for i,j, k in zip(validation_ids, label_valid_ids, sorted(formatted_indices_tr_val[1]))]

                #print("json_dict['training']", json_dict['training'])

                folds = split_into_folds(indices_tr_val[0], used_groups, args.num_folds, start_one_based_index=True) # split train indices into N folds containing the indices + 1, to correspond to the filenames!!
                for i, fold in enumerate(folds):
                    for idx in sorted(fold):
                        formatted_idx = f"{idx:03}"
                        for entry in json_dict["training"]:
                            if entry["subject_id"] == formatted_idx:
                                entry["fold"] = i
                                break

            else:   
                if args.indices: # if indices are provided, use them
                    indices = args.indices
                else:
                    # if no groups are provided, split the data randomly and assign folds to the training data
                    # Split data into training and validation based on randomly sample jj indices 
                    jj=math.ceil(len(filenames_image) * args.train_val_ratio)   # TODO: reconsider randomness and seed
                    random.seed(42) 
                    indices = random.sample(range(len(filenames_image)), jj)
                
                # Split data into training and validation based on the indices
                train_ids = [image_paths[i] for i in indices]
                validation_ids = [image_paths[i] for i in range(len(filenames_image)) if i not in indices]
                label_train_ids = [label_paths[i] for i in indices]
                label_valid_ids = [label_paths[i] for i in range(len(filenames_label)) if i not in indices]  

                # Split training data into N random folds
                fold_size = len(json_dict["training"]) // args.num_folds
            
                for i in range(args.num_folds):
                    for j in range(fold_size):
                        json_dict["training"][i * fold_size + j]["fold"] = i   
                
                # TODO: check whether this part is inside or outside the else statement
                # TODO: is random stuff still needed or can we throw it? If needed, add if modality is T1 and FLAIR or multiple modalities
                json_dict['numTraining'] = len(train_ids)
                json_dict['numValidation'] = len(validation_ids)
                json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for i, j in zip(train_ids, label_train_ids)]
                json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for i,j in zip(validation_ids, label_valid_ids)]
       

        print("os.path.join(output_folder, json_filename)", os.path.join(output_folder, json_filename))
        with open(os.path.join(output_folder, json_filename), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=True)
                print("file created at: ", os.path.join(output_folder, json_filename))
                f.close()
            

    #------------------------------------------------------------------------------------------        
    # TESTING is needed when one of the following modes is selected
    #------------------------------------------------------------------------------------------
    if args.mode in ['test', 'train_predict', 'finetune_predict']:
        print("THIS IS TESTING")
        test_dir_v0 = os.path.join(args.dataroot, 'labels', 'final')
        test_dir_v1 = os.path.join(args.dataroot, 'image_Ts')
        test_dir_v2 = os.path.join(args.dataroot, 'imagesTs')

        # set test_dir to the correct folder
        if os.path.exists(test_dir_v0):
            test_dir = test_dir_v0
        elif os.path.exists(test_dir_v1) and args.datasettype == 'ASCHOPLEX':
            test_dir = test_dir_v1
        elif os.path.exists(test_dir_v2) and args.datasettype == 'NNUNETV2':
            test_dir = test_dir_v2
        elif os.path.exists(test_dir_v2) and args.datasettype == 'UMAMBA':
            test_dir = test_dir_v2
        else:
            raise ValueError("The folder structure is not correct. Please, provide the data in the correct format.")

        
        filenames_test_image = os.listdir(test_dir)
        filenames_test_image.sort()

        # remove hidden files and .DS_Store files 
        filenames_test_image = [f for f in filenames_test_image if not f.startswith('.')]
        # List of image paths    
        test_image_paths = [os.path.join(test_dir, filename) for filename in filenames_test_image]
            
        # Check that all files have ending .nii or .nii.gz
        for i in range(len(filenames_test_image)):
            if not(filenames_test_image[i].endswith('.nii') | filenames_test_image[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")

        # Implement group logic for test data
        print("this is arggs.groups_json_path", args.groups_json_path)
        print("is None", args.groups_json_path is None)
        if args.groups_json_path:
            with open(args.groups_json_path, 'r') as f:
                groups_all = json.load(f)
                # get key index_order_test from json file if in keys else throw error
                if 'index_order_test' in groups_all.keys():
                    groups = groups_all['index_order_test']
                else:
                    print("No index_order_test in groups json file. Please, provide the correct json file.")
                    raise ValueError("No index_order_test in groups json file. Please, provide the correct json file.")
                
            if args.include_groups:
                used_groups = filter_groups(groups, args.include_groups)
            else:
                used_groups = groups

            group_distribution = calculate_group_distribution(used_groups)
            split_ratios = (1.0,)
            indices_test = split_data(used_groups, group_distribution, split_ratios)

            # Format indices to be zero-padded to three digits and add 1 to match the filenames (ESSENTIAL)
            formatted_indices_test = [f"{i+1:03}" for i in indices_test[0]]
            
            # Split data into training and validation based on the formatted indices
            test_ids = []
            test_ids2 = []
            for i in sorted(formatted_indices_test):
                for path in test_image_paths:
                    if f"{i}_" in path:
                        test_ids.append(path)
                        if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                            test_ids2.append(path.replace('0000', '0001'))
                        break
            #test_ids = [test_image_paths[i] for i in indices_test[0]]       # TODO: CONTINUE HERE!!!
            json_dict['numTest'] = len(test_ids)
            if args.modality == ['T1', 'FLAIR'] or args.modality == ['FLAIR', 'T1']:
                json_dict['testing'] = [{"image": ['%s' %i, '%s' %j], "subject_id": '%s' %k} for i, j, k in zip(test_ids, test_ids2, sorted(formatted_indices_test))]
            else:
                json_dict['testing'] = [{"image": '%s' %i, "subject_id": '%s' %k} for i, k in zip(test_ids, sorted(formatted_indices_test))]
        else:

            json_dict['numTest'] = len(test_image_paths)
            json_dict['testing'] = [{"image": '%s' %i} for i in test_image_paths] # TODO: maybe there will be an error here

            
    # create json file - manually set path to pretrained model
    if args.mode in ['finetune', 'finetune_predict']:
        # check whether the path to the pretrained model is provided
        if args.pretrained_model_path is None:
            raise ValueError("The path to the pretrained model is not provided. Please, provide the path to the pretrained model or start training from scratch using mode 'train'.")
        else: 
            json_dict['pretrained_model'] = {'path': args.pretrained_model_path}

    # if datasettype is NNUNETV2, add channel_names to the json file. 

    with open(os.path.join(output_folder, json_filename), 'w') as f: # opens file for writing and automatically close the file after block of code.
            json.dump(json_dict, f, indent=4, sort_keys=True) # writes json_dict dictionary to the file f in JSON format.
            f.close()
  
    return os.path.join(output_folder, json_filename)


def parse_modality(value):
    try:
        # Try to parse the value as a list using literal_eval
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        # If parsing fails, return the value as a string
        return value


def setup_argparse():
    parser = argparse.ArgumentParser(description="Configure and initiate training, finetuning, or testing pipeline with unified dataset path.")
    parser.add_argument("--benchmark_dataroot", type=str, required=False, help="Base path to the benchmark dataset directory")
    parser.add_argument("--dataroot", type=str, required=True, help="Base path to the dataset directory")
    parser.add_argument('--datasettype', type=str, default='ASCHOPLEX', required=False, choices=['ASCHOPLEX', 'NNUNETV2', 'UMAMBA'], help='Type of dataset (ASCHOPLEX, NNUNETV2 or UMAMBA)')
    parser.add_argument("--description", required=False, help="Data description")
    parser.add_argument('--fileending', type=str, default='.nii', required=False, help='File ending of the images')
    parser.add_argument("--groups_json_path", required=False, default='/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json', help="Path to the groups json file")  # TODO: check
    parser.add_argument("--include_groups", type=ast.literal_eval, required=False, help="List of groups to include in the experiment")    # TODO: check, maybe  type=str, nargs='+',
    parser.add_argument("--indices", type=ast.literal_eval, required=False, help="List of indices to use for training and validation")    # TODO: check, maybe  type=str, nargs='+',
    parser.add_argument('--json_dir', type=str, default=None, required=False, help='Name of the directory where the json files are stored. If nothing is specified, json will be stored in data folder, otherwise a new folder will be created where it will be created.')
    parser.add_argument('--modality', type=parse_modality, default='MR', required=False, help='Modality of the dataset', choices=['T1', 'FLAIR', 'T1xFLAIR',  ['T1', 'FLAIR'], ['FLAIR', 'T1'], 'MR', ['T1'], ['FLAIR'], ['T1xFLAIR']])
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'finetune', 'test', 'train_predict', 'finetune_predict'], default='train', help="Operation mode")
    parser.add_argument("--num_folds", type=int, required=False, default=5, help="The number of folds to split the training data into. Default is 5.")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model")
    parser.add_argument("--train_val_ratio", type=float, default=0.5 ,help="Ratio of training to validation data")
    parser.add_argument("--work_dir", required=False, help="working directory. Default is the same as dataroot")
    parser.add_argument("--skip_validation", required=False, default=True, action='store_true', help="Skip the validation data")
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    generate_json(args)

