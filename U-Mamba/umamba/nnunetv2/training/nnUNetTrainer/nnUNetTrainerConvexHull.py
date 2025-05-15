import os
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import torch
from torch import nn, autocast
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.convex_hull_transform import ConvexHullTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
    
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

from typing import Union, List, Tuple

from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation

from nnunetv2.training.dataloading.convex_data_loader_3d import nnUNetDataLoader3D_convex

class nnUNetTrainerConvexHull(nnUNetTrainer):
    """
    A custom nnUNetTrainer that applies ConvexHullTransform during training.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.margin = 5
        
   #@staticmethod # ! TODO: THIS CHANGES ERROR -> should it be static or not????? 
    
    '''
    def get_training_transforms(self, patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        print("in get_training_transforms of OUR TRAINER!!!!!!!!!")
        tr_transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data, order_resampling_seg, border_val_seg, use_mask_for_norm,
            is_cascaded, foreground_labels, regions, ignore_label
        )
        print("now add the custom convex hull transform")
        #tr_transforms = []
        
        output_folder = self.output_folder
        convex_hull_path = os.path.join(output_folder, 'convex_hull.npy')
        # Add the custom convex hull transform at the beginning
        tr_transforms.transforms.insert(0, ConvexHullTransform(convex_hull_path=convex_hull_path))
        #tr_transforms.append(ConvexHullTransform(convex_hull_path='/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull.npy'))
        
        print("tr_transforms", tr_transforms)
        print("NOW JUST APPEND OLD STUFF")
        
        return tr_transforms
    '''
    
    '''
    def get_tr_and_val_datasets(self):
        # create dataset split
        #print("in get_tr_and_val_datasets DATASETS of OUR TRAINER!!!!!!!!!")
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        
        self._compute_convex_hull(dataset_tr, margin=5)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        
        return dataset_tr, dataset_val
    '''
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        #print("in get_tr_and_val_datasets DATASETS of OUR TRAINER!!!!!!!!!")
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        
        self._compute_convex_hull(dataset_tr, margin=5)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        
        dataset_tr = self._apply_convex_hull_transform(dataset_tr)
        dataset_val = self._apply_convex_hull_transform(dataset_val)
        print("AFTER APPLYING CONVEX HULL TRANSFORM to dataset_tr and dataset_val")
        # save plot of slice each of data and seg for debugging
        #import matplotlib.pyplot as plt
        print("shape: ", dataset_tr.dataset[tr_keys[0]]['data'].shape)
        #plt.imshow(dataset_tr.dataset[tr_keys[0]]['data'][0, :, 100, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_x_dataset_tr.png')
        #plt.close()
        
        #plt.imshow(dataset_val.dataset[val_keys[0]]['data'][0, :, 100, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_x_dataset_val.png')
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
    
    '''
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            convex_hull_path = os.path.join(self.output_folder, 'convex_hull.npy')
            print("NOW READING CONVEX HULL FROM: ", convex_hull_path)
            
            dl_tr = nnUNetDataLoader3D_convex(dataset_tr, self.batch_size,
                                             initial_patch_size,
                                             self.configuration_manager.patch_size,
                                             self.label_manager,
                                             oversample_foreground_percent=self.oversample_foreground_percent,
                                             sampling_probabilities=None, pad_sides=None, 
                                             convex_hull_path=convex_hull_path)
            dl_val = nnUNetDataLoader3D_convex(dataset_val, self.batch_size,
                                              self.configuration_manager.patch_size,
                                              self.configuration_manager.patch_size,
                                              self.label_manager,
                                              oversample_foreground_percent=self.oversample_foreground_percent,
                                              sampling_probabilities=None, pad_sides=None,
                                              convex_hull_path=convex_hull_path)
        return dl_tr, dl_val
        '''
        
    def _apply_convex_hull_transform(self, dataset: nnUNetDataset) -> nnUNetDataset:
        """
        Apply Convex Hull Transform to the data and segmentation.
        Args:
            dataset (nnUNetDataset): The dataset to apply the Convex Hull Transform to.
        """
        convex_hull_path = os.path.join(self.output_folder, 'convex_hull.npy')
        convex_hull_transform = ConvexHullTransform(convex_hull_path=convex_hull_path)
        print("_APPLYING CONVEX HULL TRANSFORM")
        for key in dataset.dataset.keys():
            
            data, seg, properties = dataset.load_case(key)
            print("data shape", data.shape)
            
            # Apply ConvexHullTransform before padding
            # create data_dict to pass to ConvexHullTransform
            data_dict = {'data': data, 'seg': seg, 'properties': properties}
            data_dict_updated = convex_hull_transform(**data_dict) # To pass data_dict as keyword arguments
            print("successfully applied ConvexHullTransform")
            # Update the dataset with transformed data and segmentation
            dataset.dataset[key]['data'] = data_dict_updated['data']
            dataset.dataset[key]['seg'] = data_dict_updated['seg']
        
        return dataset
            
    '''  
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # Apply Convex Hull Transform
        data_dict = {'data': data, 'seg': target, 'properties': None}
        data_dict = self.convex_hull_transform(**data_dict)
        data = data_dict['data']
        target = data_dict['seg']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    '''
    
    
    def _compute_convex_hull(self, dataset: nnUNetDataset, margin: int):
        # make convex_hull_path plan and fold adaptive
        #conve
        output_folder = self.output_folder
        

        convex_hull_path = os.path.join(output_folder, 'convex_hull.npy')

         
        if os.path.exists(convex_hull_path):
            print("Convex hull already computed, skipping...")
            return
        segmentations = [np.load(dataset.dataset[key]['data_file'])['seg'] for key in dataset.dataset.keys()]
        
        # pad all segmentations to the same shape (max shape)
        max_shape_z = np.max([s.shape[1] for s in segmentations])
        max_shape_y = np.max([s.shape[2] for s in segmentations])
        max_shape_x = np.max([s.shape[3] for s in segmentations])
        
        segmentations_padded = []
        for s in segmentations:
            pad_z = max_shape_z - s.shape[1]
            pad_y = max_shape_y - s.shape[2]
            pad_x = max_shape_x - s.shape[3]
            
            pad_width = ((0,0),
                         (pad_z // 2, pad_z - pad_z // 2),  # Per la profondità
                         (pad_y // 2, pad_y - pad_y // 2),  # Per l'altezza
                         (pad_x // 2, pad_x - pad_x // 2))   # Per la larghezza
                         
            segmentations_padded.append(np.pad(s, pad_width=pad_width, mode='constant', constant_values=0).squeeze(0))

        segmentations_padded = np.stack(segmentations_padded)
           
     
        
        convex_hull = self._create_volume_within_convex_hull(segmentations_padded)
        
        np.save(convex_hull_path, convex_hull)
        os.chmod(convex_hull_path, 0o777)
        
        
    def _create_convex_hull(self, segmentations):
        
        combined_seg = np.sum(segmentations, axis=0) > 0
        points = np.argwhere(combined_seg)
      

        hull = ConvexHull(points)
        return points, hull

    def _create_volume_within_convex_hull(self, segmentations):                                 
        points, hull = self._create_convex_hull(segmentations)
        delunay = Delaunay(points[hull.vertices])

        volume = np.zeros(segmentations.shape[1:], dtype=np.float32)
        # Create a mask with ones inside the convex hull and zeros outside
        for x in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                for z in range(volume.shape[2]):
                    # Find the simplices containing the given points. 
                    # returned integers in the array are the indices of the simplex the corresponding point is in. If -1 is returned, the point is in no simplex
                    if delunay.find_simplex([x, y, z]) >= 0:
                        volume[x, y, z] = 1.0
        
        # Apply dilation to add margin
        volume_with_margin = binary_dilation(volume, iterations=self.margin) * 1.0

        return volume_with_margin
    

class nnUNetTrainerConvexHullUMambaBot(nnUNetTrainerConvexHull):

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
        
        print("UMambaBot: {}".format(model))

        return model
            
            
            
class nnUNetTrainerConvexHullUMambaEnc(nnUNetTrainerConvexHull):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                dataset_json,
                                configuration_manager: ConfigurationManager,
                                num_input_channels,
                                enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                        num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model

