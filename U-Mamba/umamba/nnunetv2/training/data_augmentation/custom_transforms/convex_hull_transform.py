import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation

class ConvexHullTransform(AbstractTransform):
    """
    A custom data augmentation transform that applies a convex hull mask to the input data.
    Args:
        convex_hull_path (str): Path to the numpy file containing the convex hull mask.
        margin (int, optional): Margin to be applied around the convex hull. Default is 5.
        p_per_sample (float, optional): Probability of applying the transform per sample. Default is 1.0.
    Methods:
        __call__(**data_dict):
            Applies the convex hull mask to the input data.
            Args:
                data_dict (dict): Dictionary containing the input data and segmentation masks.
                    - 'data' (numpy.ndarray): The input data to be transformed.
                    - 'seg' (numpy.ndarray): The segmentation masks.
                    - 'properties' (dict): Additional properties of the data.
            Returns:
                dict: The transformed data dictionary with the convex hull mask applied to the 'data' key.
    """
    
    def __init__(self, convex_hull_path, margin=5, p_per_sample=1.0):
        self.convex_hull = np.expand_dims(np.load(convex_hull_path), axis=(0,1))


    def __call__(self, **data_dict):
        #print("ConvexHullTransform")
        
        data = data_dict['data']
        
        # save the plot of one data slice for debugging
        import matplotlib.pyplot as plt
        #plt.imshow(data[0, 0, 100])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_x.png')
        #plt.close()
        
        #plt.imshow(data[0, :, 100, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_z.png')
        #plt.close()
        
        #plt.imshow(data[0, 100, :, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_slice_y.png')
        #plt.close()
        
        
        #print("Data shape:", data.shape) # TODO: how can Datashape be (2, 1, 238, 196, 208) when max size of volume is 240x240x180??? 
        #print("Convex hull shape:", self.convex_hull.shape)
        data_shape = data.shape
        hull_shape = self.convex_hull.shape[2:]  # Ignore batch and channel dimensions
        #print("HULL SHAPE: ", hull_shape)
        z_crop = (hull_shape[0] - data_shape[1]) // 2
        y_crop = (hull_shape[1] - data_shape[2]) // 2
        x_crop = (hull_shape[2] - data_shape[3]) // 2

        # Ensure non-negative crop values
        z_crop = max(0, z_crop)
        y_crop = max(0, y_crop)
        x_crop = max(0, x_crop)

        #print(f"Cropping values: z={z_crop}, y={y_crop}, x={x_crop}")

        # Adjust slices to avoid negative indices
        z_slice = slice(z_crop, hull_shape[0] - z_crop) if hull_shape[0] - 2*z_crop == data_shape[1] else slice(z_crop, data_shape[1] + z_crop)
        y_slice = slice(y_crop, hull_shape[1] - y_crop) if hull_shape[1] - 2*y_crop == data_shape[2] else slice(y_crop, data_shape[2] + y_crop)
        x_slice = slice(x_crop, hull_shape[2] - x_crop) if hull_shape[2] - 2*x_crop == data_shape[3] else slice(x_crop, data_shape[3] + x_crop)

        #print(f"Adjusted slices: z={z_slice}, y={y_slice}, x={x_slice}")

        # Ensure that convex_hull shape matches data shape exactly
        convex_hull = self.convex_hull[:, :, z_slice, y_slice, x_slice].squeeze(0)
        #print("Convex hull shape after squeezing:", convex_hull.shape)
     
        
        #print("new convex_hull_shape = ", convex_hull.shape)
        #print("data_shape = ", data.shape)  
        #print("convex_hull[2:].shape = ", convex_hull[1:].shape)
        #print("data[1:].shape = ", data[1:].shape)
        #print("SHapes match: ", convex_hull[1:].shape == data[1:].shape)
        
        # As data is read only, we need to deepcopy it
        data_x_convex_hull = np.copy(data)
        data_x_convex_hull *= convex_hull
        
        # plot and save the transformed data slice for debugging
        #plt.imshow(data_x_convex_hull[0, :, 100, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_x_convex_hull_slice_y.png')
        #plt.close()
        
        #plt.imshow(data_x_convex_hull[0, 120, :, :])
        #plt.savefig('/var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/convex_hull_transform_data_x_convex_hull_slice_z.png')
        #plt.close()
        
        # Mask the input data
        #data *= convex_hull
        
        data_dict['data'] = data_x_convex_hull
       
        return data_dict

    
