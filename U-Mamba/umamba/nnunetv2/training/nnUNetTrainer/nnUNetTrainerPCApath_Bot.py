import os
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import shutil
import torch
from torch import nn, autocast
from torch import distributed as dist
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import nibabel as nib
from time import time

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

from nnunetv2.nets.UMambaBot_3d_copy import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.nets.UMambaEnc_3d_copy import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

from typing import Union, List, Tuple

from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation

from nnunetv2.training.dataloading.convex_data_loader_3d import nnUNetDataLoader3D_convex

class nnUNetTrainerPCApath(nnUNetTrainer):
    """
    A custom nnUNetTrainer that applies ConvexHullTransform during training.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.margin = 5
        
   #@staticmethod # ! : THIS CHANGES ERROR -> should it be static or not????? 
    
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        #print("in get_tr_and_val_datasets DATASETS of OUR TRAINER!!!!!!!!!")
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        
        # save plot of slice each of data and seg for debugging
        #import matplotlib.pyplot as plt
        # print keys of dataset_tr and dataset_val
        #print("dataset_tr keys: ", dataset_tr.keys())
        #print("shape: ", dataset_tr.dataset[tr_keys[0]]['data'].shape)
        #plt.imshow(dataset_tr.dataset[tr_keys[0]]['data'][0, :, 100, :])
        #plt.savefig('/data1/user/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_x_dataset_tr.png')
        #plt.close()
        
        #plt.imshow(dataset_val.dataset[val_keys[0]]['data'][0, :, 100, :])
        #plt.savefig('/data1/user/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_x_dataset_val.png')
        #plt.close()
        
        return dataset_tr, dataset_val
    
    
    def get_dataloaders(self):
        # ! this happens BEFOR train_step function in run_training of nnUNetTrainer
        """
        Prepares and returns the training and validation data loaders with appropriate transformations and augmentations.
        This method determines the dimensionality of the data (2D or 3D) based on the patch size and configures the necessary
        data augmentation and transformation pipelines for both training and validation datasets. It also handles the 
        creation of multi-threaded or single-threaded data loaders based on the allowed number of processes for data 
        augmentation.
        Returns:
            tuple: A tuple containing:
                - mt_gen_train: The training data loader with transformations and augmentations applied.
                - mt_gen_val: The validation data loader with transformations applied.
        """
        
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        #print("in nnUNetTrainerConvexHull get_dataloaders BEFORE get_plain_dataloaders")
        # ! Here we changed the position!
        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim) # ! CHANGED POSITION HERE! 
        
        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else: # ! Goes into LimitedLenWrapper!!
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val
    
        
        
    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        # Inject PCA scan vector if available
        # print attributes of self.network
        print(hasattr(self.network, 'mamba_layer'))
        print("zaaaaaa")

        # 
        if hasattr(self.network, 'mamba_layer'): #and getattr(self.network, 'mamba_scan_type', '') == 'pca':
            print("[PCA] Attempting to load PCA scan vector...")
            pca_vector_path = os.path.join(self.output_folder, 'pca_scan_vector.npy')
            if os.path.exists(pca_vector_path):
                try:
                    pca_vector = np.load(pca_vector_path)
                    self.network.mamba_layer.set_scan_vector(pca_vector)
                    self.print_to_log_file("[PCA] Scan vector successfully loaded and set.")
                except Exception as e:
                    self.print_to_log_file(f"[PCA] Failed to load PCA vector: {e}")
            else:
                self.print_to_log_file("[PCA] No PCA scan vector found yet. Using fallback scan order.")
        else:
            print("[PCA] Network does not have a mamba layer or PCA scan type is not set. Using fallback scan order.")
            
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

    def on_train_epoch_end2(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        
        # Adding pca for scan path #! user 
        if self.current_epoch == 0 and self.local_rank == 0:
            print("Computing PCA scan path from ground truth masks...")

        dataset_tr, _ = self.get_tr_and_val_datasets()
        all_masks = []

        for key in dataset_tr.keys():
            _, seg, _ = dataset_tr.load_case(key)  # seg shape: (1, D, H, W)
            # print shape of seg
            print(f"Shape of segmentation for key {key}: {seg.shape}")
            seg = seg[0]  # assume binary mask
            all_masks.append(seg)
            # print size of all_masks
            print(f"Size of all_masks after appending key {key}: {len(all_masks)}")

        # Find max shape along each dimension
        shapes = [mask.shape for mask in all_masks]
        max_shape = np.max(np.array(shapes), axis=0)

        def pad_central(arr, target_shape):
            pad_width = []
            for i in range(len(target_shape)):
                total_pad = target_shape[i] - arr.shape[i]
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))
            return np.pad(arr, pad_width, mode='constant', constant_values=0)

        all_masks = [pad_central(mask, max_shape) for mask in all_masks]
        all_masks = np.stack(all_masks)  # shape: (N, D, H, W)
        # print shape of all_masks
        # print(f"Shape of all_masks: {all_masks.shape}")
        mean_mask = np.mean(all_masks, axis=0)

        coords = np.argwhere(mean_mask > 0.05) # shape (N, 3), gives positions
        
        weights = mean_mask[mean_mask > 0.05] # shape (), gives actual values

        if len(coords) > 0:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(coords)

            principal_vector = pca.components_[0]
            np.save(os.path.join(self.output_folder, "pca_scan_vector.npy"), principal_vector)
            # unit vector (array of length 3) representing main direction of variance among  coords in mask region
            print(f"PCA vector saved to: {self.output_folder}/pca_scan_vector.npy")
        else:
            print("No foreground voxels found in mean mask.")

    ## second version of it 
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        # --- PCA scan path logic ---
        if self.current_epoch == 0 and self.local_rank == 0:
            print("Computing PCA scan path from ground truth masks...")

            dataset_tr, _ = self.get_tr_and_val_datasets()
            model = self.network
            model.eval()
            stem_block = model.encoder.stem.to(self.device)

            # Find max shape (once, before the loop)
            max_shape = None
            for key in dataset_tr.keys():
                _, seg, _ = dataset_tr.load_case(key)
                seg = seg.astype(np.float32)
                seg_tensor = torch.from_numpy(seg).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    stem_out = stem_block(seg_tensor).squeeze(0)
                if max_shape is None:
                    max_shape = stem_out.shape
                else:
                    max_shape = tuple(np.maximum(max_shape, stem_out.shape))
                del seg_tensor, stem_out
                torch.cuda.empty_cache()
    
            # Now accumulate, padding as needed
            mean_feature_map = None
            num_cases = 0
            for key in dataset_tr.keys():
                _, seg, _ = dataset_tr.load_case(key)
                seg = seg.astype(np.float32)
                seg_tensor = torch.from_numpy(seg).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    stem_out = stem_block(seg_tensor).squeeze(0)
                # Pad to max_shape
                pad = [(0, m - s) for s, m in zip(stem_out.shape, max_shape)]
                stem_out = torch.nn.functional.pad(stem_out, [p for tup in reversed(pad) for p in tup])
                if mean_feature_map is None:
                    mean_feature_map = stem_out.clone()
                else:
                    mean_feature_map += stem_out
                num_cases += 1
                del seg_tensor, stem_out
                torch.cuda.empty_cache()
            mean_feature_map /= num_cases
            mean_projection = mean_feature_map.max(dim=0)[0]  # (D, H, W)

            # Convert to numpy
            mean_np = mean_projection.cpu().numpy()

            # Save the projected map for debugging
            mean_nifti = nib.Nifti1Image(mean_np, np.eye(4))
            nib.save(mean_nifti, os.path.join(self.output_folder, "mean_projection.nii.gz"))
            print(f"[PCA] Saved mean projection to: {self.output_folder}/mean_projection.nii.gz")

            # Get coordinates of high-activation regions
            coords = np.argwhere(mean_np > 0.05)

            if len(coords) > 0:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(coords)
                principal_vector = pca.components_[0]
                # Save PCA vector
                np.save(os.path.join(self.output_folder, "pca_scan_vector.npy"), principal_vector)

                # Create an empty volume for the scan line
                scan_line = np.zeros_like(mean_np, dtype=np.uint8)

                # Center point of the structure
                center = coords.mean(axis=0)

                # Vector scaling
                length = min(mean_np.shape) * 0.9  # 90% of the shortest axis
                start = center - 0.5 * length * principal_vector
                end = center + 0.5 * length * principal_vector

                # Interpolate 3D line (50 points)
                for i in np.linspace(0, 1, 50):
                    pt = (1 - i) * start + i * end
                    pt_round = np.round(pt).astype(int)
                    if np.all((pt_round >= 0) & (pt_round < mean_np.shape)):
                        scan_line[tuple(pt_round)] = 1

                # Optional: thicken the line (makes it easier to see)
                scan_line = binary_dilation(scan_line, iterations=1).astype(np.uint8)

                # Save as NIfTI
                scan_affine = np.eye(4)  # change this if you have a real affine
                scan_nifti = nib.Nifti1Image(scan_line, scan_affine)
                nib.save(scan_nifti, os.path.join(self.output_folder, "scan_vector_overlay.nii.gz"))

                print(f"[PCA] Scan line overlay saved to: {self.output_folder}/scan_vector_overlay.nii.gz")

                
                print(f"[PCA] Scan vector saved to: {self.output_folder}/pca_scan_vector.npy")
            else:
                print("[PCA] No meaningful foreground voxels found.")


   
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                dataset_json,
                                configuration_manager: ConfigurationManager,
                                num_input_channels,
                                enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model

