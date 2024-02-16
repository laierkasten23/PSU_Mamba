import os
import argparse
import random
import sys
import json
from collections import OrderedDict


class WriteTrainJSON:
    """
    Class for writing .json files to run from training from scratch, finetuning and/or the prediction of Choroid Plexus segmentations.

    """
    def __init__(self, dataroot: str=".", description=None, work_dir: str=".", train_json: str="train.json"):
        """
        Initializes the class with the given parameters.

        :param dataroot: The path to the data directory. (/var/data/MONAI_Choroid_Plexus/dataset_monai)
        :param description: The description of the experiment.
        :param work_dir: The working directory. (/var/data/student_home/lia/thesis/monai_segmentation/monai_training)
        :param train_json: The name of the train.json file. (train.json)
        """
        self.dataroot = dataroot
        if description is None:
            self.description='Dataset for Choroid Plexus segmentation'
        elif isinstance(description, str):
            self.description=description
        self.work_dir = work_dir
        self.train_json = train_json
        self.file=[]

    def write_train_json(self, json_filename: str="train.json"):

        # Set path to output file
        output_folder = os.path.join(self.work_dir, 'JSON_dir')

        # Create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_dir = self.dataroot
        label_dir = os.path.join(self.dataroot, 'labels', 'final')

        filenames_image = os.listdir(image_dir)
        filenames_image.sort()
        # Check if the label directory is in filenames_image and remove it from the list of filenames
        if 'labels' in filenames_image:
            filenames_image.remove('labels')

        filenames_label = os.listdir(label_dir)
        filenames_label.sort()   

        image_paths = [os.path.join(image_dir, filename) for filename in filenames_image]
        label_paths = [os.path.join(label_dir, filename) for filename in filenames_label]

        if len(filenames_image)!=len(filenames_label):
                raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")
        
        
        # create json file - manually set
        json_dict = OrderedDict()
        json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
        json_dict['description'] = self.description
        json_dict['tensorImageSize'] = "3D"
        json_dict['modality'] = {
            "0": "MR"
        }
            
        json_dict['labels'] = {
            "0": "background",
            "1": "Choroid Plexus"
        }

        json_dict['numTraining'] = len(image_paths)

        json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_paths, image_paths)]
        
        # Randomise training data
        random.seed(42)
        random.shuffle(json_dict["training"])
        
        # Split training data into N random folds
        num_folds = 5
        fold_size = len(json_dict["training"]) // num_folds
        for i in range(num_folds):
            for j in range(fold_size):
                json_dict["training"][i * fold_size + j]["fold"] = i

        with open(os.path.join(output_folder, json_filename), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=True)




class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

# Main
if __name__ == '__main__':
    print('Starting launching_tool :)')

    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Pipeline for training selected model from scratch or finetuning with N subjects with selected pretrained models"
    )

    # Add the parameters positional/optional
    parser.add_argument('--dataroot', required=True, default="/var/data/MONAI_Choroid_Plexus/dataset_train_from_scratch_monai" , help="Data directory. Where the data is stored")
    parser.add_argument('--description', required=False, help="Data description")
    parser.add_argument('--work_dir', required=True, default="/var/data/student_home/lia/thesis/monai_segmentation/monai_training", help="working directory")
    parser.add_argument('--train_json', required=False, default="train.json", help="Name of the train.json file")
    # Parse the arguments
    args = parser.parse_args()
    print(args)
 
    print('Writing JSON file for training.....')
    json_file=WriteTrainJSON(args.dataroot, args.description, args.work_dir).write_train_json(json_filename=args.train_json)

