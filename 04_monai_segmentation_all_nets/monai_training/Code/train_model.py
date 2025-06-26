import os
import shutil
import sys
from copy import deepcopy
from typing import Any, Dict, List
from monai.apps.utils import get_logger
from monai.apps.auto3dseg import DataAnalyzer
from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.apps.auto3dseg.utils import export_bundle_algo_history
from monai.auto3dseg.algo_gen import AlgoGen
from monai.auto3dseg.utils import algo_to_pickle
from monai.bundle.config_parser import ConfigParser
from monai.config import print_config
from monai.utils import ensure_tuple
from monai.utils.enums import AlgoKeys # added to work

from project_dir.monai_segmentation.monai_training.Code.bundle_generation_script import MyBundleGen
print_config()

join=os.path.join

logger = get_logger(module_name=__name__)

class TrainModel:
    """
    Class for training a model from scratch.

    Args:
        work_dir (str): The working directory for the model training. Default is the current directory.
        dataroot (str): The root directory for the data. Default is the current directory.
        json_file (str): The path to the JSON file containing the data list. Default is the current directory.
        output_dir (str): The output directory for the trained model. If not provided, a default directory will be created.

    Attributes:
        work_dir (str): The working directory for the model training.
        Dataroot (str): The root directory for the data.
        JSON_file (str): The path to the JSON file containing the data list.
        output_dir (str): The output directory for the trained model.

    Methods:
        training_run: Runs the training process.

    """

    def __init__(self, work_dir: str=".", dataroot: str = ".", json_file: str = ".", output_dir=None):
        """
        Initializes the TrainModel object.

        Args:
            work_dir (str): The working directory for the model training. Default is the current directory.
            dataroot (str): The root directory for the data. Default is the current directory.
            json_file (str): The path to the JSON file containing the data list. Default is the current directory.
            output_dir (str): The output directory for the trained model. If not provided, a default directory will be created.
        """
        self.work_dir=work_dir
        self.dataroot=dataroot
        self.json_file=json_file
        if output_dir is None:
            self.output_dir=join(self.work_dir, 'working_directory_training')   
        elif  isinstance(output_dir, str):
            self.output_dir=output_dir
    
    def training_run(self):
        """
        Runs the training process.

        Returns:
            str: The output directory for the trained model.
        """
        dataroot = self.dataroot
        work_dir = self.output_dir

        # create working directory
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        algorithm_path=join(os.environ['MONAIDIR'], 'DNN_models', 'algorithm_templates_yaml') 

        da_output_yaml = join(work_dir, "datastats.yaml")
        data_src_cfg = join(work_dir, "data_src_cfg.yaml")

        if not os.path.isdir(dataroot):
            os.makedirs(dataroot)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        # write to a json file
        datalist = self.json_file

        # 1. Analyze Dataset
        da = DataAnalyzer(datalist, dataroot, output_path=da_output_yaml)
        da.get_all_case_stats()

        data_src = {
            "modality": "MRI",
            "datalist": datalist,
            "dataroot": dataroot,
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)

        # 2. Training

        bundle_generator = MyBundleGen(
            algo_path=work_dir, 
            algo_template=algorithm_path,
            data_stats_filename=da_output_yaml,
            data_src_cfg_name=data_src_cfg
        )

        # Get parsed content for the first algorithm
        parsed_content = bundle_generator.algos[0].get_parsed_content()
        print(parsed_content)

        bundle_generator.generate(work_dir, num_fold=1) # : why only one fold here? What does make sense?
        history = bundle_generator.get_history()
        ("DONE")
        export_bundle_algo_history(history)


        max_epochs = 100 # : what makes sense here? 
        max_epochs = max(max_epochs, 2)

        train_param = {
            "CUDA_VISIBLE_DEVICES": [0],  # use only 1 gpu
            "num_iterations": 10000,
            "num_iterations_per_validation": 2 * max_epochs, # : what makes sense here?
            "num_images_per_batch": 1,
            "num_epochs": max_epochs,
            "num_warmup_iterations": 2 * max_epochs,
        }

        for h in history: # [1::4] (added) to get only the algo_bunde objects
            for i, (_, algo) in enumerate(h.items()):
                if i % 2 == 1: 
                    algo.train(train_param)
                    print("DONE")
        
        return self.output_dir


