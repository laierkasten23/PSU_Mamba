import os
import argparse
import json
import random
import math
from collections import OrderedDict

# https://github.com/FAIR-Unipd/ASCHOPLEX/blob/main/Code/create_json.py
# and https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/notebooks/msd_datalist_generator.ipynb 


def generate_json(args):
    """
    Class for writing .json files to run training/ finetuning/ testing/ training and predicting or finetuning and predicting Choroid Plexus segmentations.

    dataroot = "/var/data/MONAI_Choroid_Plexus/dataset_monai_train_from_scratch"
    work_dir = "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training"
    json_file=WriteTrainJSON(dataroot, work_dir).write_train_val_json(json_filename = "train_val3.json")

    Usage: 
        python step2_create_json.py --mode "train" --dataroot "/var/data/MONAI_Choroid_Plexus/dataset_monai_train_from_scratch" --work_dir "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training" --train_val_ratio 0.5 --num_folds 5   
        python step2_create_json.py --mode "test" --dataroot "/var/data/MONAI_Choroid_Plexus/dataset_aschoplex" --work_dir "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training" --train_val_ratio 0.5 --num_folds 5   

    Args:
        dataroot (str): The root directory of the dataset. Default is ".".
        description (str, optional): Description of the dataset. Default is None.
        mode (str): One of: 'train', 'finetune', 'test', 'train_predict', 'finetune_predict'. Which operation mode is used. 
        work_dir (str): The working directory. Default is ".".
        train_val_ratio (float): The ratio of training data to validation data. Default is 0.5.
        num_folds (int): The number of folds to split the training data into. Default is 5.
        pretrained_model_path (str): The path to the pretrained model. Default is None.
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
                # TODO: maybe also incorporate the other folder structure
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

        where a, b, i, j are subject identifiers.  
        """


        # Set path to output file
    output_folder = os.path.join(args.work_dir, 'JSON_dir')

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Set name of the json file
    if args.mode in ['train', 'finetune']:
        json_filename="dataset_train_val.json"
    elif args.mode in ['train_predict', 'finetune_predict']:
        json_filename="dataset_train_val_pred.json"
    else:
        json_filename="dataset_test.json"


    # create json file - manually set: general information
    #------------------------------------------------------------------------------------------
    json_dict = OrderedDict()
    json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
    json_dict['description'] = args.description if args.description is not None else "Dataset for Choroid Plexus segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['modality'] = {
        "0": "MR"
    }
            
    json_dict['labels'] = {
        "0": "background",
        "1": "Choroid Plexus"
    }

    # training and validation are needed when one of the following modes is selected
    #------------------------------------------------------------------------------------------
    if args.mode in ['train', 'finetune', 'train_predict', 'finetune_predict']:
        # Check the folder structure
        if os.path.exists(os.path.join(args.dataroot, 'labels')):
            image_dir = args.dataroot
            label_dir = os.path.join(args.dataroot, 'labels', 'final')
        elif os.path.exists(os.path.join(args.dataroot, 'image_Tr')):
            image_dir = os.path.join(args.dataroot, 'image_Tr')
            label_dir = os.path.join(args.dataroot, 'label_Tr')
        else:
            raise ValueError("The folder structure is not correct. Please, provide the data in the correct format.")

        filenames_image = os.listdir(image_dir)
        filenames_image.sort()

        # Check if the label directory is in filenames_image and remove it from the list of filenames
        if 'labels' in filenames_image:
            filenames_image.remove('labels')
        
        filenames_label = os.listdir(label_dir)
        filenames_label.sort()   

        # remove hidden files and .DS_Store files 
        filenames_image = [f for f in filenames_image if not f.startswith('.')]
        filenames_label = [f for f in filenames_label if not f.startswith('.')]

        if len(filenames_image)!=len(filenames_label):
            raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")

        # List of image and label paths
        image_paths = [os.path.join(image_dir, filename) for filename in filenames_image]
        label_paths = [os.path.join(label_dir, filename) for filename in filenames_label]

        # Check that all files have ending .nii or .nii.gz
        for i in range(len(filenames_image)):
            if not(filenames_image[i].endswith('.nii') | filenames_image[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
            if not(filenames_label[i].endswith('.nii') | filenames_label[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                
        # Split data into training and validation based on randomly sample jj indices 
        jj=math.ceil(len(filenames_image) * args.train_val_ratio) 
        random.seed(42) 
        indices = random.sample(range(len(filenames_image)), jj)

        train_ids = [image_paths[i] for i in indices]
        validation_ids = [image_paths[i] for i in range(len(filenames_image)) if i not in indices]
        label_train_ids = [label_paths[i] for i in indices]
        label_valid_ids = [label_paths[i] for i in range(len(filenames_label)) if i not in indices]     

        json_dict['numTraining'] = len(train_ids)
        json_dict['numValidation'] = len(validation_ids)
        json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for i, j in zip(train_ids, label_train_ids)]
        json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for i,j in zip(validation_ids, label_valid_ids)]

        random.seed(42)
        random.shuffle(json_dict["training"])

        # Split training data into N random folds
        fold_size = len(json_dict["training"]) // args.num_folds
        for i in range(args.num_folds):
            for j in range(fold_size):
                json_dict["training"][i * fold_size + j]["fold"] = i

        print("os.path.join(output_folder, json_filename)", os.path.join(output_folder, json_filename))
        with open(os.path.join(output_folder, json_filename), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=True)
                print("file created at: ", os.path.join(output_folder, json_filename))
                f.close()
            
            
    # testing is needed when one of the following modes is selected
    #------------------------------------------------------------------------------------------
    if args.mode in ['test', 'train_predict', 'finetune_predict']:
        test_dir = os.path.join(args.dataroot, 'image_Ts')
        # TODO: maybe also incorporate the other folder structure: 
        # But there, the testting images are all the images in the main folder, 
        # #which do not have a label in the labels -> final folder 
        
        filenames_test_image = os.listdir(test_dir)
        filenames_test_image.sort()

        # remove hidden files and .DS_Store files 
        filenames_test_image = [f for f in filenames_test_image if not f.startswith('.')]
            
        test_image_paths = [os.path.join(test_dir, filename) for filename in filenames_test_image]
            
            
        for i in range(len(filenames_test_image)):
            if not(filenames_test_image[i].endswith('.nii') | filenames_test_image[i].endswith('.nii.gz')):
                raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
            
        json_dict['numTest'] = len(test_image_paths)
        json_dict['testing'] = [{"image": '%s' %i} for i in filenames_test_image] # TODO: maybe there will be an error here

            
    # create json file - manually set path to pretrained model
    if args.mode in ['finetune', 'finetune_predict']:
        # check whether the path to the pretrained model is provided
        if args.pretrained_model_path is None:
            raise ValueError("The path to the pretrained model is not provided. Please, provide the path to the pretrained model or start training from scratch using mode 'train'.")
        else: 
            json_dict['pretrained_model'] = {'path': args.pretrained_model_path}

    with open(os.path.join(output_folder, json_filename), 'w') as f: # opens file for writing and automatically close the file after block of code.
            json.dump(json_dict, f, indent=4, sort_keys=True) # writes json_dict dictionary to the file f in JSON format.
            f.close()
  
    return os.path.join(output_folder, json_filename)





def setup_argparse():
    parser = argparse.ArgumentParser(description="Configure and initiate training, finetuning, or testing pipeline with unified dataset path.")
    parser.add_argument("--description", required=False, help="Data description")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'finetune', 'test', 'train_predict', 'finetune_predict'], default='train', help="Operation mode")
    parser.add_argument("--dataroot", type=str, required=True, help="Base path to the dataset directory")
    parser.add_argument("--work_dir", required=True, help="working directory")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model")
    parser.add_argument("--train_val_ratio", type=float, default=0.5 ,help="Ratio of training to validation data")
    parser.add_argument("--num_folds", type=int, required=False, default=5, help="The number of folds to split the training data into. Default is 5.")

    return parser.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    generate_json(args)

