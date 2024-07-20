import os
import nibabel as nib
import numpy as np
import json
import argparse

import monai
print(monai.__version__)

from monai.apps.auto3dseg import AutoRunner, export_bundle_algo_history, import_bundle_algo_history
from monai.utils.enums import AlgoKeys
from monai.auto3dseg import algo_to_pickle
from monai.config import print_config
from monai.bundle.config_parser import ConfigParser

def get_directory_names(path):
    """Return a list of directory names in the given path."""
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]



def run_autorunner(work_dir: str="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/monai_segmentation/monai_training/working_directory_0606", 
                   dataroot: str="/home/linuxlia/Lia_Masterthesis/data/pazienti", 
                   json_path: str="/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training/JSON_dir/train_val.json",
                   algos: list = ["resnet3d", "unet3d", "vnet3d"],
                   templates_path_or_url: str = "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/monai_segmentation/DNN_models/algorithm_templates_yaml/"):
    """
    This function runs the AutoRunner from MONAI to train the models

    :param work_dir: Working directory for the AutoRunner where to save the results
    :param dataroot: path to the dataset
    :param json_path: path to the json file containing the train and val subjects
    :param algos: list of algorithms to be used
    :param templates_path_or_url: path to the algorithm templates

    Example Usage:

    """

    data_src = {
        "modality": "MRI",
        "datalist": json_path, #datalist, # give path to json file, it is not necessary to already read the json file
        "dataroot": dataroot,
    }


    data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")
    ConfigParser.export_config_file(data_src, data_src_cfg)

    # Check whether all the algos are available


    runner = AutoRunner(work_dir=work_dir, input=data_src_cfg, algos=algos, templates_path_or_url=templates_path_or_url)
    runner.run()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='.', help="Working directory for the AutoRunner where to save the results")
    parser.add_argument('--dataroot', type=str, default='/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP')
    parser.add_argument('--json_path', type=str, default='/home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json', help="Path to the json file containing the train and val subjects")
    parser.add_argument('--algos', nargs='+', default="all", help="List of algorithms to be used or 'all'/'none' to use all algorithms")
    parser.add_argument('--templates_path_or_url', type=str, default="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/02_monai_segmentation/DNN_models/algorithm_templates_yaml/", help="Path to the algorithm templates")
    args = parser.parse_args()

    # Dynamically set default algos based on directories in templates_path_or_url
    default_algos = get_directory_names(args.templates_path_or_url)
    
    # Use the default algos if 'all' or 'none' is specified, or use the provided list
    if not args.algos or args.algos == ['all'] or args.algos == ['none']:
        algos_to_use = default_algos
    else:
        algos_to_use = args.algos


    run_autorunner(args.work_dir, args.dataroot, args.json_path, algos=args.algos, templates_path_or_url=args.templates_path_or_url)
    print('Autorunner done')
    
    # python step3_run_AutoRunner.py --work_dir /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/01_aschoplex_from_scratch/working_directory_0807 --dataroot /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP --json_path /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json --algos all --templates_path_or_url /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/01_aschoplex_from_scratch/DNN_models/algorithm_templates/
    # python step3_run_AutoRunner.py --work_dir /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/02_monai_segmentation/monai_training/working_directory_0807 --dataroot /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP --json_path /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json --algos all --templates_path_or_url /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/02_monai_segmentation/DNN_models/algorithm_templates_yaml/
    # python step3_run_AutoRunner.py --work_dir /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/02_monai_segmentation/monai_training/working_directory_0807 --dataroot /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP --json_path /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json --algos DynUnet128dice --templates_path_or_url /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/02_monai_segmentation/DNN_models/algorithm_templates_yaml/