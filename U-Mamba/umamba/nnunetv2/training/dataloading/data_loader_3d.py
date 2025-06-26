import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

import matplotlib.pyplot as plt # Added by Lia 


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)

        
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            # print('SHAPE (generate_train_batch -> here it is still correct):', shape) # ! HERE DATA IS OKAY; WTF IS THE DATA SHAPE ABOVE?????
            
            # Added by Lia to show if images are consistent. Shape is (z,y,x)
            # plt.imshow(data[0,100,:,:])
            # plt.gca().invert_yaxis()
            # plt.savefig(f"/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/visualization/patch{i}{j}.png")
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            #print("data shape after cropping", data.shape)
            #print("GO TO HELL, nnUNetDataLoader3D generate_train_batch")
            
            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            
            # shape here is still 3 dim
            

            # ! THIS IS DONE BEFORE DATA AUGMENTATION WHICH MEANS THAT OUR CONVEX HULL CALCULATION NEEDS TO BE DONE HERE!
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # Here the cropped and padded patch id added to the data_all and seg_all arrays (LIA) 
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            
            # (batch, C, X, Y, Z)
            
            # Added by Lia: 
            # Plot some slices of the patch
            #self.plot_slices(data_all[j])
            #################

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

    def plot_slices(self, patch):
        # Plot the middle slice of each dimension
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        mid_slices = [patch.shape[i] // 2 for i in range(1, 4)]
        
        axes[0].imshow(patch[0, mid_slices[0], :, :], cmap='gray')
        axes[0].set_title('Axial Slice')
        
        axes[1].imshow(patch[0, :, mid_slices[1], :], cmap='gray')
        axes[1].set_title('Coronal Slice')
        
        axes[2].imshow(patch[0, :, :, mid_slices[2]], cmap='gray')
        axes[2].set_title('Sagittal Slice')
        
        plt.show()

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    folder = '/data1/LIA/Umamba_data/nnUNet_preprocessed/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetPlans_3d_fullres'
    folder = ''
    ds = nnUNetDataset(folder)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
