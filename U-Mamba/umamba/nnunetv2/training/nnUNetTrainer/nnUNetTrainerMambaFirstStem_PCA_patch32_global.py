import os
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import shutil
import torch
from torch import nn, autocast
from torch import distributed as dist
import torch.nn.functional as F  

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import nibabel as nib
from time import time
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from sklearn.decomposition import PCA

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.collate_outputs import collate_outputs 

from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.nets.UMambaFirst_global32 import get_umamba_first_3d_from_plans

from typing import Union, List, Tuple

from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation

from nnunetv2.training.dataloading.convex_data_loader_3d import nnUNetDataLoader3D_convex
from nnunetv2.nets.UMambaFirst import MambaLayer

class nnUNetTrainerMambaFirstStem_PCA_patch32_global(nnUNetTrainer):
    """
    A custom nnUNetTrainer that applies ConvexHullTransform during training.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.margin = 5
        os.environ["UMAMBA_OUTPUT_FOLDER"] = self.output_folder
    
    # For global PCA
    def on_train_start(self):
        super().on_train_start()
        if self.scan_type == "global_pca":
            # Compute global PCA vectors here
            self.global_pca_vector = compute_pca_over_training_set(...) #TODO implement this method to compute the global PCA vector
            self.network.set_global_pca_vector(self.global_pca_vector) # TODO implement this method in the network class


        


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if self.current_epoch == 0:
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
            
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    # For local PCA
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        
        # Adding pca for scan path #! Lia 
        if self.current_epoch == 0 and self.local_rank == 0:
            print("Computing PCA scan path from ground truth masks...")

        dataset_tr, _ = self.get_tr_and_val_datasets()
        all_masks = []

        if self.scan_type == "local_pca":
            print("Updating local PCA vectors from current predictions...")
            
            # Aggregate prediction maps over validation/training set
            prediction_map = self.get_mean_prediction_map()
            
            # Recompute local PCA vectors
            self.local_pca_vectors = compute_local_pca(
                prediction_map=prediction_map,
                patch_coords=self.patch_coords,
                patch_size=self.pca_patch_size
            )

        # Update model
        self.network.set_pca_vectors(self.local_pca_vectors)
        
        
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                dataset_json,
                                configuration_manager: ConfigurationManager,
                                num_input_channels,
                                enable_deep_supervision: bool = True) -> nn.Module:
        

        output_folder = os.environ.get("UMAMBA_OUTPUT_FOLDER")
        print(f"UMamba output folder: {output_folder}")
        
        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_first_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision, 
                                        output_folder=output_folder)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaEnc: {}".format(model))
        return model