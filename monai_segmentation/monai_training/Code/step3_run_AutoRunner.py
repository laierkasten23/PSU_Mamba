import os
import nibabel as nib
import numpy as np
import json
import argparse

from monai.apps.auto3dseg import AutoRunner, export_bundle_algo_history, import_bundle_algo_history
from monai.utils.enums import AlgoKeys
from monai.auto3dseg import algo_to_pickle
from monai.config import print_config
from monai.bundle.config_parser import ConfigParser




def run_autorunner(work_dir: str, 
                   dataroot: str="/var/data/MONAI_Choroid_Plexus/dataset_monai_train_from_scratch", 
                   json_path: str="/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training/JSON_dir/train_val.json",
                   algos: list = ["resnet3d", "unet3d", "vnet3d"],
                   templates_path_or_url: str = "/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/DNN_models/algorithm_templates_yaml/"):
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
    print('Starting reorganizing Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='.', help="Working directory for the AutoRunner where to save the results")
    parser.add_argument('--dataroot', type=str, default='/var/data/MONAI_Choroid_Plexus/ANON_DATA_01_labels')
    parser.add_argument('--json_path', type=str, default='/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/monai_training/JSON_dir/train_val.json', help="Path to the json file containing the train and val subjects")
    parser.add_argument('--algos', type=list, default=["resnet3d", "unet3d", "vnet3d"], help="List of algorithms to be used")
    parser.add_argument('--templates_path_or_url', type=str, default="/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/DNN_models/algorithm_templates_yaml/", help="Path to the algorithm templates")
    args = parser.parse_args()

    run_autorunner(args.work_dir, args.dataroot, args.json_path, algos=["resnet3d", "unet3d", "vnet3d"], templates_path_or_url="/var/data/student_home/lia/phuse_thesis_2024/monai_segmentation/DNN_models/algorithm_templates_yaml/")
    print('Reorganizing Dataset finished')
    