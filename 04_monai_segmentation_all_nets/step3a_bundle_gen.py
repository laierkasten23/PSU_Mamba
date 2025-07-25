# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compusernce with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping

import torch

from monai.apps import download_and_extract
from monai.apps.utils import get_logger
from monai.auto3dseg.algo_gen import Algo, AlgoGen
from monai.auto3dseg.utils import algo_to_pickle
from monai.bundle.config_parser import ConfigParser
from monai.utils import ensure_tuple

logger = get_logger(module_name=__name__)
ALGO_HASH = os.environ.get("MONAI_ALGO_HASH", "d7bf36c")

__all__ = ["userBundleAlgo", "BundleGen"]





class userBundleAlgo(Algo):
    """
    An algorithm represented by a set of bundle configurations and scripts.

    ``BundleAlgo.cfg`` is a ``monai.bundle.ConfigParser`` instance.

    .. code-block:: python

        from monai.apps.auto3dseg import BundleAlgo

        data_stats_yaml = "/workspace/data_stats.yaml"
        algo = BundleAlgo(template_path=../algorithms/templates/segresnet2d/configs)
        algo.set_data_stats(data_stats_yaml)
        # algo.set_data_src("../data_src.json")
        algo.export_to_disk(".", algo_name="segresnet2d_1")

    This class creates MONAI bundles from a directory of 'bundle template'. Different from the regular MONAI bundle
    format, the bundle template may contain placeholders that must be filled using ``fill_template_config`` during
    ``export_to_disk``. Then created bundle keeps the same file structure as the template.

    """

    def __init__(self, template_path: str):
        """
        Create an Algo instance based on the predefined Algo template.

        Args:
            template_path: path to the root of the algo template.

        """

        self.template_path = template_path
        self.data_stats_files = ""
        self.data_list_file = ""
        self.output_path = ""
        self.name = ""
        self.best_metric = None
        # track records when filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}
        self.fill_records: dict = {}

    def set_data_stats(self, data_stats_files: str):
        """
        Set the data analysis report (generated by DataAnalyzer).

        Args:
            data_stats_files: path to the datastats yaml file
        """
        self.data_stats_files = data_stats_files

    def set_data_source(self, data_src_cfg: str):
        """
        Set the data source configuration file

        Args:
            data_src_cfg: path to a configuration file (yaml) that contains datalist, dataroot, and other params.
                The config will be in a form of {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot":
                "path_dir_data"}
        """
        self.data_list_file = data_src_cfg

    def fill_template_config(self, data_stats_filename: str, algo_path: str, **kwargs) -> dict:
        """
        The configuration files defined when constructing this Algo instance might not have a complete training
        and validation pipelines. Some configuration components and hyperparameters of the pipelines depend on the
        training data and other factors. This API is provided to allow the creation of fully functioning config files.
        Return the records of filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}.

        Args:
            data_stats_filename: filename of the data stats report (generated by DataAnalyzer)

        Notes:
            Template filling is optional. The user can construct a set of pre-filled configs without replacing values
            by using the data analysis results. It is also intended to be re-implemented in subclasses of BundleAlgo
            if the user wants their own way of auto-configured template filling.
        """
        return {}

    def export_to_disk(self, output_path: str, algo_name: str, **kwargs):
        """
        Fill the configuration templates, write the bundle (configs + scripts) to folder `output_path/algo_name`.

        Args:
            output_path: Path to export the 'scripts' and 'configs' directories.
            algo_name: the identifier of the algorithm (usually contains the name and extra info like fold ID).
            kwargs: other parameters, including: "copy_dirs=True/False" means whether to copy the template as output
                instead of inplace operation, "fill_template=True/False" means whether to fill the placeholders
                in the template. other parameters are for `fill_template_config` function.

        """
        if kwargs.pop("copy_dirs", True):
            self.output_path = os.path.join(output_path, algo_name)
            os.makedirs(self.output_path, exist_ok=True)
            if os.path.isdir(self.output_path):
                shutil.rmtree(self.output_path)
            shutil.copytree(self.template_path, self.output_path)
        else:
            self.output_path = self.template_path
        if kwargs.pop("fill_template", True):
            self.fill_records = self.fill_template_config(self.data_stats_files, self.output_path, **kwargs)
        logger.info(self.output_path)

    def _create_cmd(self, train_params=None):
        """
        Create the command to execute training.

        """
        if train_params is not None:
            params = deepcopy(train_params)

        train_py = os.path.join(self.output_path, "scripts", "train.py")
        config_dir = os.path.join(self.output_path, "configs")

        if os.path.isdir(config_dir):
            base_cmd = ""
            for file in os.listdir(config_dir):
                if len(base_cmd) == 0:
                    base_cmd += f"{train_py} run --config_file="
                else:
                    base_cmd += ","  # Python Fire does not accept space
                # Python Fire may be confused by single-quoted WindowsPath
                config_yaml = Path(os.path.join(config_dir, file)).as_posix()
                base_cmd += f"'{config_yaml}'"

        if "CUDA_VISIBLE_DEVICES" in params:
            devices = params.pop("CUDA_VISIBLE_DEVICES")
            n_devices, devices_info = len(devices), ",".join([str(x) for x in devices])
        else:
            n_devices, devices_info = torch.cuda.device_count(), ""
        if n_devices > 1:
            cmd = f"torchrun --nnodes={1:d} --nproc_per_node={n_devices:d} "
        else:
            cmd = "python "  # : which system python?
        cmd += base_cmd
        if params and isinstance(params, Mapping):
            for k, v in params.items():
                cmd += f" --{k}={v}"
        return cmd, devices_info

    def _run_cmd(self, cmd: str, devices_info: str):
        """
        Execute the training command with target devices information.

        """
        try:
            logger.info(f"Launching: {cmd}")
            ps_environ = os.environ.copy()
            if devices_info:
                ps_environ["CUDA_VISIBLE_DEVICES"] = devices_info
            normal_out = subprocess.run(cmd.split(), env=ps_environ, check=True, capture_output=True)
            logger.info(repr(normal_out).replace("\\n", "\n").replace("\\t", "\t"))
        except subprocess.CalledProcessError as e:
            output = repr(e.stdout).replace("\\n", "\n").replace("\\t", "\t")
            errors = repr(e.stderr).replace("\\n", "\n").replace("\\t", "\t")
            raise RuntimeError(f"subprocess call error {e.returncode}: {errors}, {output}") from e
        return normal_out

    def train(self, train_params=None):
        """
        Load the run function in the training script of each model. Training parameter is predefined by the
        algo_config.yaml file, which is pre-filled by the fill_template_config function in the same instance.

        Args:
            train_params:  to specify the devices using a list of integers: ``{"CUDA_VISIBLE_DEVICES": [1,2,3]}``.
        """
        cmd, devices_info = self._create_cmd(train_params)
        return self._run_cmd(cmd, devices_info)

    def get_score(self, *args, **kwargs):
        """
        Returns validation scores of the model trained by the current Algo.
        """
        config_yaml = os.path.join(self.output_path, "configs", "hyper_parameters.yaml")
        parser = ConfigParser()
        parser.read_config(config_yaml)
        ckpt_path = parser.get_parsed_content("ckpt_path", default=self.output_path)

        dict_file = ConfigParser.load_config_file(os.path.join(ckpt_path, "progress.yaml"))
        # dict_file: a list of scores saved in the form of dict in progress.yaml
        return dict_file[-1]["best_avg_dice_score"]  # the last one is the best one

    def get_inferer(self, *args, **kwargs):
        """
        Load the InferClass from the infer.py. The InferClass should be defined in the template under the path of
        `"scripts/infer.py"`. It is required to define the "InferClass" (name is fixed) with two functions at least
        (``__init__`` and ``infer``). The init class has an override kwargs that can be used to override parameters in
        the run-time optionally.

        Examples:

        .. code-block:: python

            class InferClass
                def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, **override):
                    # read configs from config_file (sequence)
                    # set up transforms
                    # set up model
                    # set up other hyper parameters
                    return

                @torch.no_grad()
                def infer(self, image_file):
                    # infer the model and save the results to output
                    return output

        """
        infer_py = os.path.join(self.output_path, "scripts", "infer.py")
        if not os.path.isfile(infer_py):
            raise ValueError(f"{infer_py} is not found, please check the path.")

        config_dir = os.path.join(self.output_path, "configs")
        configs_path = [os.path.join(config_dir, f) for f in os.listdir(config_dir)]

        spec = importlib.util.spec_from_file_location("InferClass", infer_py)
        infer_class = importlib.util.module_from_spec(spec)
        sys.modules["InferClass"] = infer_class
        spec.loader.exec_module(infer_class)
        return infer_class.InferClass(configs_path, *args, **kwargs)

    def predict(self, predict_params=None):
        """
        Use the trained model to predict the outputs with a given input image. Path to input image is in the params
        dict in a form of {"files", ["path_to_image_1", "path_to_image_2"]}. If it is not specified, then the
        prediction will use the test images predefined in the bundle config.

        Args:
            predict_params: a dict to override the parameters in the bundle config (including the files to predict).

        """
        if predict_params is None:
            params = {}
        else:
            params = deepcopy(predict_params)

        files = params.pop("files", ".")
        inferer = self.get_inferer(**params)
        return [inferer.infer(f) for f in ensure_tuple(files)]

    def get_output_path(self):
        """Returns the algo output paths to find the algo scripts and configs."""
        return self.output_path


# path to download the algo_templates
default_algo_zip = (
    f"https://github.com/Project-MONAI/research-contributions/releases/download/algo_templates/{ALGO_HASH}.tar.gz"
)

# default algorithms
default_algos = {
    "segresnet2d": dict(_target_="segresnet2d.scripts.algo.Segresnet2dAlgo", template_path="segresnet2d"),
    "dints": dict(_target_="dints.scripts.algo.DintsAlgo", template_path="dints"),
    "swinunetr": dict(_target_="swinunetr.scripts.algo.SwinunetrAlgo", template_path="swinunetr"),
    "segresnet": dict(_target_="segresnet.scripts.algo.SegresnetAlgo", template_path="segresnet"),
}

default_algos = {
    "UNETR_DiceCE_0": dict(_target_="UNETR_128_DiceCE_0.scripts.algo.UNETRAlgo", template_path="UNETR_128_DiceCE_0"),
}


class userBundleGen(AlgoGen):
    """
    This class generates a set of bundles according to the cross-validation folds, each of them can run independently.

    Args:
        algo_path: the directory path to save the algorithm templates. Default is the current working dir.
        algos: if dictionary, it outlines the algorithm to use. if None, automatically download the zip file
            from the default link. if string, it represents the download link.
            The current default options are released at:
            https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg
        data_stats_filename: the path to the data stats file (generated by DataAnalyzer)
        data_src_cfg_name: the path to the data source config YAML file. The config will be in a form of
            {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot": "path_dir_data"}

    .. code-block:: bash

        python -m monai.apps.auto3dseg BundleGen generate --data_stats_filename="../algorithms/data_stats.yaml"
    """

    def __init__(self, algo_path: str = ".", algos=None, data_stats_filename=None, data_src_cfg_name=None):
        self.algos: Any = []

        if algos is None or isinstance(algos, str):
            # trigger the download process
            zip_download_dir = TemporaryDirectory()
            algo_compressed_file = os.path.join(zip_download_dir.name, "algo_templates.tar.gz")
            #download_and_extract(default_algo_zip if algos is None else algos, algo_compressed_file, algo_path)
            zip_download_dir.cleanup()
            sys.path.insert(0, os.path.join(algo_path, "algorithm_templates"))
            algos = deepcopy(default_algos)
            for name in algos:
                algos[name]["template_path"] = os.path.join(
                    algo_path, "algorithm_templates", algos[name]["template_path"]
                )

        if isinstance(algos, dict):
            for algo_name, algo_params in algos.items():
                try:
                    self.algos.append(ConfigParser(algo_params).get_parsed_content())
                except RuntimeError as e:
                    if "ModuleNotFoundError" in str(e):
                        msg = """Please make sure the folder structure of an Algo Template follows
                            [algo_name]
                            ├── configs
                            │   ├── hyperparameters.yaml  # automatically generated yaml from a set of ``template_configs``
                            │   ├── network.yaml  # automatically generated network yaml from a set of ``template_configs``
                            │   ├── transforms_train.yaml  # automatically generated yaml to define transforms for training
                            │   ├── transforms_validate.yaml  # automatically generated yaml to define transforms for validation
                            │   └── transforms_infer.yaml  # automatically generated yaml to define transforms for inference
                            └── scripts
                                ├── test.py
                                ├── __init__.py
                                └── validate.py
                        """
                        raise RuntimeError(msg) from e
                self.algos[-1].name = algo_name
        else:
            self.algos = ensure_tuple(algos)

        self.data_stats_filename = data_stats_filename
        self.data_src_cfg_filename = data_src_cfg_name
        self.history: List[Dict] = []

    def set_data_stats(self, data_stats_filename: str):
        """
        Set the data stats filename

        Args:
            data_stats_filename: filename of datastats
        """
        self.data_stats_filename = data_stats_filename

    def get_data_stats(self):
        """Get the filename of the data stats"""
        return self.data_stats_filename

    def set_data_src(self, data_src_cfg_filename):
        """
        Set the data source filename

        Args:
            data_src_cfg_filename: filename of data_source file
        """
        self.data_src_cfg_filename = data_src_cfg_filename

    def get_data_src(self):
        """Get the data source filename"""
        return self.data_src_cfg_filename

    def get_history(self) -> List:
        """get the history of the bundleAlgo object with their names/identifiers"""
        return self.history

    def generate(self, output_folder=".", num_fold: int = 5):
        """
        Generate the bundle scripts/configs for each bundleAlgo

        Args:
            output_folder: the output folder to save each algorithm.
            num_fold: the number of cross validation fold
        """
        fold_idx = list(range(num_fold))
        for algo in self.algos:
            for f_id in ensure_tuple(fold_idx):
                data_stats = self.get_data_stats()
                data_src_cfg = self.get_data_src()
                gen_algo = deepcopy(algo)
                gen_algo.set_data_stats(data_stats)
                gen_algo.set_data_source(data_src_cfg)
                name = f"{gen_algo.name}_{f_id}"
                gen_algo.export_to_disk(output_folder, name, fold=f_id)
                algo_to_pickle(gen_algo, template_path=algo.template_path)
                self.history.append({name: gen_algo})  # track the previous, may create a persistent history
