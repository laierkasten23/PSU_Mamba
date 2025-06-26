import numpy as np
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.data_augmentation.custom_transforms.convex_hull_transform import ConvexHullTransform

class nnUNetDataLoader3D_convex(nnUNetDataLoader3D):
    def __init__(self, *args, convex_hull_path, **kwargs): # : just added
        super().__init__(*args, **kwargs)
        self.convex_hull_path = convex_hull_path
        
    def generate_train_batch(self):
        #print("WE ARE IN nnUNetDataLoader3D_convex generate_train_batch")
        
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        
        convex_hull_transform = ConvexHullTransform(convex_hull_path=self.convex_hull_path)
        
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(i)
            
            # Apply ConvexHullTransform before padding
            # create data_dict to pass to ConvexHullTransform
            data_dict = {'data': data, 'seg': seg, 'properties': properties}
            data_dict_updated = convex_hull_transform(**data_dict) # To pass data_dict as keyword arguments
            
            data = data_dict_updated['data']
            seg = data_dict_updated['seg']
            case_properties.append(properties)
            
            
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            
            
            
            # unpac dict again
            data = data_dict_updated['data']
            #print("data shape after ConvexHullTransform", data.shape)
            
            
            # Cropping
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            seg = seg[this_slice]
            #print("data shape after cropping", data.shape)

            
            #print("data shape after ConvexHullTransform", data.shape)

            # Padding
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            #print("data shape after padding", data_all[j].shape)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties}