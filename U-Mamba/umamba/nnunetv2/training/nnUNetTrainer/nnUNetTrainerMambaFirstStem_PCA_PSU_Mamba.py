import os
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import shutil
import torch
from torch import nn, autocast
from torch import distributed as dist
import torch.nn.functional as F  
from sklearn.decomposition import PCA

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import nibabel as nib
from time import time
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from sklearn.decomposition import PCA
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
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
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.pca_utils import get_patch_coords, compute_local_pca, compute_global_pca, extract_patch, extract_patches_and_origins
from torch.nn.parallel import DistributedDataParallel as DDP

from torch._dynamo import OptimizedModule

from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.nets.UMambaFirst_PCA import get_umamba_first_3d_from_plans #! network which is used

from typing import Union, List, Tuple

from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation

from nnunetv2.training.dataloading.convex_data_loader_3d import nnUNetDataLoader3D_convex
from nnunetv2.nets.UMambaFirst import MambaLayer

class nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba(nnUNetTrainer):
    """
    A custom nnUNetTrainer that generally applies PCA to the stem features of the model.
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device('cuda'), **kwargs):
        # Extract custom args and REMOVE them from kwargs before passing to base class
        self.scan_type = kwargs.pop('scan_type', 'global_pca') # kwargs.pop('scan_type', 'global_pca') # : 'global_pca', 'local_pca', 'x', 'y', 'z', 'xy_scan', 'diag'
        self.pca_patch_size = kwargs.pop('pca_patch_size', None)


        # Ensure patch size is tuple
        if self.pca_patch_size is not None:
            if isinstance(self.pca_patch_size, (list, np.ndarray)):
                self.pca_patch_size = tuple(self.pca_patch_size)
            elif isinstance(self.pca_patch_size, int):
                self.pca_patch_size = (self.pca_patch_size,) * 3

        # Build output folder name based on scan_type and pca_patch_size
        scan_str = self.scan_type
        if self.scan_type == 'local_pca' and self.pca_patch_size is not None:
            patch_str = "_".join(str(i) for i in self.pca_patch_size)
            scan_str += f"_patch{patch_str}"
        self.output_folder_base = os.path.join(
            nnUNet_results, plans['dataset_name'],
            self.__class__.__name__ + '__' + plans['plans_name'] + "__" + configuration + f"_{scan_str}") \
                if nnUNet_results is not None else None
        os.environ["UMAMBA_OUTPUT_FOLDER"] = self.output_folder_base
        
        # Now call base class only with remaining kwargs
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, **kwargs)


        self.local_pca_vectors = None
        self.local_pca_coords = None
        self.global_pca_vector = None
        
        
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                self.enable_deep_supervision,
                self.scan_type,
                self.pca_patch_size if self.scan_type == 'local_pca' else None,
                self.local_pca_vectors if self.scan_type == 'local_pca' else None,
                self.local_pca_coords if self.scan_type == 'local_pca' else None,
                self.global_pca_vector if self.scan_type == 'global_pca' else None,
                self.output_folder
            ).to(self.device)
            print("net initialized with scan type:", self.scan_type, "and pca patch size:", self.pca_patch_size)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)

        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        self.plot_network_architecture()
        self._save_debug_information()
        
        # Determin global PCA vectors if scan_type is global_pca
        if self.scan_type == "global_pca" or self.scan_type == "local_pca":
            self.print_to_log_file("Computing global and local PCA vectors using stem features of GT...")
            
            # -- Efficient mask loading --
            dataset_tr, _ = self.get_tr_and_val_datasets()
            all_masks = []
            
            for key in dataset_tr.keys():
                _, seg, _ = dataset_tr.load_case(key)  # seg: (1, D, H, W)
                all_masks.append(seg[0])  # drop channel dim
                
            shapes = [m.shape for m in all_masks]
            max_shape = np.max(np.stack(shapes), axis=0)

            def pad_centered(arr, target_shape):
                return np.pad(arr, [( (t - s) // 2, (t - s + 1) // 2 ) for s, t in zip(arr.shape, target_shape)],
                            mode='constant', constant_values=0)

            all_masks = np.stack([pad_centered(m, max_shape) for m in all_masks])  # (N, D, H, W)
            mean_mask = np.mean(all_masks, axis=0)  # (D, H, W)
            binary_mask_np = (mean_mask > 0.05).astype(np.uint8)

            # Save binary mask as NIfTI
            import nibabel as nib
            affine = np.eye(4)
            binary_mask_nifti = nib.Nifti1Image(binary_mask_np, affine)
            output_path = os.path.join(self.output_folder, "global_pca_binary_mask.nii.gz")
            nib.save(binary_mask_nifti, output_path)
            self.print_to_log_file(f"[PCA] Saved binary mask to {output_path}")

            # -- Pass binary mask through stem --
            binary_mask_tensor = torch.from_numpy(binary_mask_np).float().to(self.device)
            if self.num_input_channels > 1:
                binary_mask_tensor = binary_mask_tensor.repeat(self.num_input_channels, 1, 1, 1).unsqueeze(0)  # (1, C, D, H, W)
            else:
                binary_mask_tensor = binary_mask_tensor.unsqueeze(0)  # (1, 1, D, H, W)
            print(f"[PCA] Binary mask shape: {binary_mask_tensor.shape}")
            
            # Pass binary mask through stem to extract features
            self.network.eval()
            with torch.no_grad():
                stem_feats = self.network.extract_stem_features(binary_mask_tensor)  # (B, C, D, H, W)

            # -- Perform PCA on spatial distribution of stem features --
            stem_feats = stem_feats.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (D, H, W, C)
            binary_coords = np.argwhere(binary_mask_np > 0)  # (N, 3)
        
            if self.scan_type == "global_pca":
                if len(binary_coords) > 0:
                    # Extract corresponding features
                    
                    pca = PCA(n_components=1)
                    pca.fit(binary_coords)

                    pca_vecs = pca.components_  # (2, 3)
                    self.global_pca_vectors = torch.from_numpy(pca_vecs).float()
                    self.network.set_global_pca_vectors(self.global_pca_vectors.to(self.device))

                    np.save(os.path.join(self.output_folder, "pca_feature_vectors.npy"), pca_vecs)
                    self.print_to_log_file("[PCA] PCA vectors computed from stem features and saved.")
                    self.print_to_log_file(f"[PCA] PCA vector shape: {pca_vecs.shape}")
                else:
                    self.print_to_log_file("[PCA] No foreground voxels found for PCA on stem features.")
                    
                    
                    
            if self.scan_type == "local_pca":
                if self.pca_patch_size is None:
                    self.pca_patch_size = (16, 16, 16)  # Default patch size if not provided
                    self.print_to_log_file(f"[PCA] Using default patch size: {self.pca_patch_size}")
                # patch the input stem features
                self.print_to_log_file("[PCA] Extracting patches and origins from stem features...")
                binary_mask_tensor_np = binary_mask_np.astype(np.uint8)[None, None]  # Add batch and channel dimensions # (1, 1, 240, 240, 180)
                print(f"[PCA] Binary mask tensor shape: {binary_mask_tensor_np.shape}")
                # Extract patches and their coordinates
                patches, coords = extract_patches_and_origins(binary_mask_tensor_np, self.pca_patch_size)

                all_vectors = []
                all_coords = []

                for patch, coord in zip(patches, coords):
                    #print("Shape of patch:", patch.shape)
                    patch = patch.squeeze(0)
                    #print("Shape of patch after squeeze:", patch.shape)
                    # Get the 3D coordinates of foreground voxels within this patch
                    foreground_coords = np.argwhere(patch > 0)  # shape: (N, 3) (why (0, 3))??? 
        
                    
                    if foreground_coords.shape[0] < 4:
                        continue
                    #print("Shape of foreground_coords:", foreground_coords.shape[0])
                    
                    pca = PCA(n_components=1)
                    try:
                        pca.fit(foreground_coords)
                        vector = torch.tensor(pca.components_[0], dtype=torch.float32)
                        vector = vector / (vector.norm() + 1e-8)
                    except Exception as e:
                        self.print_to_log_file(f"[PCA] PCA failed for local patch at {coord}: {e}")
                        continue

                    all_vectors.append(vector) # Shape: (3,)
                    #print("Shape of vector:", vector.shape)
                    #print("Shape of coord:", coord)
                    all_coords.append(coord) # Shape: (3,)

                if all_vectors:
                    self.local_pca_vectors = torch.stack(all_vectors).squeeze().to(self.device)
                    self.local_pca_coords = torch.tensor(all_coords).long().to(self.device)
                    self.network.set_local_pca_vectors(self.local_pca_vectors, self.local_pca_coords)
                    self.network.set_pca_patch_size(self.pca_patch_size)
                    self.print_to_log_file(f"[PCA] Set {len(all_vectors)} coordinate-based local PCA vectors.")
                else:
                    self.print_to_log_file("[PCA] No valid local patches found.")
            
            
    def on_train_epoch_end2(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)

        if self.scan_type != "local_pca":
            return

        self.print_to_log_file("Computing local PCA vectors from model predictions...")

        all_vectors = []
        all_coords = []


        self.network.eval()
        with torch.no_grad():
            for batch in self.dataloader_train:
                data = batch['data'].to(self.device)
                output = self.network(data)
                # If deep supervision is enabled, only use the highest resolution output
                if self.enable_deep_supervision:
                    output = output[0]

                # Use sigmoid and threshold for binary prediction
                prediction = (torch.sigmoid(output) > 0.5).float()  # Always define it

                # Optional: convert to int if regions are used
                if self.label_manager.has_regions:
                    prediction = prediction.long()

                # Optional: expand to multi-channel if needed for stem
                if self.num_input_channels > 1:
                    prediction = prediction.repeat(1, self.network.num_input_channels, 1, 1, 1)

                print(f"[PCA] Prediction shape: {prediction.shape}")
                feats = self.network.extract_stem_features(prediction)
                print(f"[PCA] Stem features shape: {feats.shape}")
                #stem_feats = stem_feats.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (D, H, W, C)
                #coords = np.argwhere(binary_mask_np > 0)  # (N, 3)
                patches, coords = extract_patches_and_origins(feats, self.pca_patch_size)

                for patch, coord in zip(patches, coords):
                    patch = patch.detach().cpu()
                    c, d, h, w = patch.shape
                    features = patch.view(c, -1).T

                    if features.shape[0] < 10:
                        continue

                    pca = PCA(n_components=2)
                    try:
                        pca.fit(features)
                        vectors = torch.tensor(pca.components_[:2], dtype=torch.float32)
                        vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)
                    except:
                        vectors = torch.eye(2, c)[:2]

                    all_vectors.append(vectors)
                    all_coords.append(coord)

        if all_vectors:
            self.local_pca_vectors = torch.stack(all_vectors).to(self.device)
            self.local_pca_coords = torch.tensor(all_coords).long().to(self.device)
            self.network.set_local_pca_vectors(self.local_pca_vectors, self.local_pca_coords)
            self.print_to_log_file(f"Set {len(all_vectors)} PCA vectors for Mamba layers.")
            
        self.network.set_local_pca_vectors(self.local_pca_vectors, self.local_pca_coords)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)

        if self.scan_type != "local_pca":
            return

        self.print_to_log_file("[PCA] Updating local PCA vectors from saved binary mask...")

        # === Load binary mask ===
        import nibabel as nib
        mask_path = os.path.join(self.output_folder, "global_pca_binary_mask.nii.gz")
        if not os.path.exists(mask_path):
            self.print_to_log_file(f"[PCA] Mask file not found: {mask_path}")
            return

        binary_mask_nifti = nib.load(mask_path)
        binary_mask_np = binary_mask_nifti.get_fdata().astype(np.uint8)
        binary_mask_np = (binary_mask_np > 0).astype(np.uint8)[None, None]  # Add batch and channel dimensions # (1, 1, 240, 240, 180)
        print(f"[PCA] Binary mask shape TRAIN EPOCH END: {binary_mask_np.shape}")

        # === Convert to tensor and prepare shape ===
        binary_mask_tensor = torch.from_numpy(binary_mask_np).float().to(self.device)
  
        self.network.eval()
        with torch.no_grad():
            feats = self.network.extract_stem_features(binary_mask_tensor)  # (1, C, D, H, W)

        # === Extract patches ===
        self.print_to_log_file(f"[PCA] Extracting local patches from stem features...")
        patches, coords = extract_patches_and_origins(feats, self.pca_patch_size)

        all_vectors = []
        all_coords = []

        for patch, coord in zip(patches, coords):
            patch = patch.squeeze(0)  # (C, D, H, W)
            mask_patch = binary_mask_np[
                coord[0]:coord[0]+self.pca_patch_size[0],
                coord[1]:coord[1]+self.pca_patch_size[1],
                coord[2]:coord[2]+self.pca_patch_size[2]
            ]

            foreground_coords = np.argwhere(mask_patch > 0)  # (N, 3)

            if foreground_coords.shape[0] < 4:
                continue

            try:
                pca = PCA(n_components=1)
                pca.fit(foreground_coords)
                vector = torch.tensor(pca.components_[0], dtype=torch.float32)
                vector = vector / (vector.norm() + 1e-8)
            except Exception as e:
                self.print_to_log_file(f"[PCA] PCA failed for local patch at {coord}: {e}")
                continue

            all_vectors.append(vector)
            all_coords.append(coord)

        # === Store updated PCA vectors ===
        if all_vectors:
            self.local_pca_vectors = torch.stack(all_vectors).to(self.device)
            self.local_pca_coords = torch.tensor(all_coords).long().to(self.device)
            self.network.set_local_pca_vectors(self.local_pca_vectors, self.local_pca_coords)
            self.print_to_log_file(f"[PCA] Updated local PCA vectors from binary mask.")
        else:
            self.print_to_log_file("[PCA] No valid patches found in binary mask.")



    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        mamba_layer = self.network.encoder.stem[1]
        print("Scan type:", self.scan_type)
        if self.scan_type in ("local_pca", "global_pca"):
            print("Found MambaLayer:", mamba_layer is not None)
            if mamba_layer is not None:
                print("[PCA] Attempting to load PCA scan vector...")
                if self.scan_type == "local_pca":
                    vectors_path = os.path.join(self.output_folder, "local_pca_vectors.npy")
                    coords_path = os.path.join(self.output_folder, "local_pca_coords.npy")
                    if os.path.exists(vectors_path) and os.path.exists(coords_path):
                        vectors = np.load(vectors_path)
                        coords = np.load(coords_path)
                        mamba_layer.set_local_pca_vectors(vectors, coords)
                        self.print_to_log_file("[PCA] Local PCA vectors loaded.")
                    else:
                        self.print_to_log_file("[PCA] Local PCA vectors not found.")
                elif self.scan_type == "global_pca":
                    pca_vector_path = os.path.join(self.output_folder, 'pca_feature_vectors.npy')
                    print(f"[PCA] Looking for global PCA vector at {pca_vector_path}")
                    if os.path.exists(pca_vector_path):
                        try:
                            pca_vector = np.load(pca_vector_path)
                            mamba_layer.set_global_pca_vectors(pca_vector)
                            self.print_to_log_file("[PCA] Global PCA vector successfully loaded and set.")
                        except Exception as e:
                            self.print_to_log_file(f"[PCA] Failed to load global PCA vector: {e}")
                    else:
                        self.print_to_log_file("[PCA] No global PCA vector found yet. Using fallback scan order.")
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

        current_epoch = self.current_epoch
        if self.current_epoch == 0:
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
            
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)

        if self.scan_type == "local_pca":
            if hasattr(self, "local_pca_vectors") and hasattr(self, "local_pca_coords"):
                np.save(os.path.join(self.output_folder, "local_pca_vectors.npy"),
                        self.local_pca_vectors.detach().cpu().numpy())
                np.save(os.path.join(self.output_folder, "local_pca_coords.npy"),
                        self.local_pca_coords.detach().cpu().numpy())
                self.print_to_log_file("Saved local PCA vectors and coordinates.")
                
        elif self.scan_type == "global_pca":
            print("self.global_pca_vector BEFORE SAVING:", self.global_pca_vector)
            if hasattr(self.network, "global_pca_vector") and self.global_pca_vector is not None:
                np.save(os.path.join(self.output_folder, "global_pca_vector.npy"),
                        self.global_pca_vector.detach().cpu().numpy())
                self.print_to_log_file("Saved global PCA vector.")
            else:
                self.print_to_log_file("Global PCA vector is None, not saving.")
                

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.print_to_log_file("Checkpoint loaded and PCA vectors (if available) restored.")
    
    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True,
            scan_type: str = 'x',
            pca_patch_size=None,
            local_pca_vectors=None,
            local_pca_coords=None,
            global_pca_vector=None,
            output_folder=None
        ) -> nn.Module:
        
        output_folder = os.environ.get("UMAMBA_OUTPUT_FOLDER")
        print(f"UMamba output folder: {output_folder}")
        
        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_first_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision, 
                                        output_folder=output_folder, 
                                        scan_type=scan_type,
                                        pca_patch_size=pca_patch_size,
                                        local_pca_vectors=local_pca_vectors,
                                        local_pca_coords=local_pca_coords,
                                        global_pca_vector=global_pca_vector)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaEnc: {}".format(model))
        return model