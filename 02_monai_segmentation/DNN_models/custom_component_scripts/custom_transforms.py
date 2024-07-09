import torch
import numpy as np
import nibabel as nib

class CropMRI(torch.nn.Module):
    def __init__(self, vol_size=[128, 128, 128]):
        super(CropMRI, self).__init__()
        self.vol_size = vol_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Get the data
        img_data = image.get_fdata()
        label_data = label.get_fdata()

        # From the centre of the entire volume, crop the volume to vol_size
        x, y, z = img_data.shape
        xx, yy, zz = label_data.shape

        # Check if the image and label have the same size
        if x != xx or y != yy or z != zz:
            raise ValueError('The image and label do not have the same size.')

        # Compute the coordinates for the cropped volume and shift it up in the x-direction
        x_start = int(x/2 - self.vol_size[0]/2) 
        x_end = int(x/2 + self.vol_size[0]/2) 
        y_start = int(y/2 - self.vol_size[1]/2) - 16
        y_end = int(y/2 + self.vol_size[1]/2) - 16
        z_start = int(z/2 - self.vol_size[2]/2) + 32
        z_end = int(z/2 + self.vol_size[2]/2) + 32

        # Crop the image and label
        img_data = img_data[x_start:x_end, y_start:y_end, z_start:z_end]
        label_data = label_data[x_start:x_end, y_start:y_end, z_start:z_end]

        # Convert the numpy arrays to PyTorch tensors
        img_data = torch.from_numpy(img_data)
        label_data = torch.from_numpy(label_data)

        return {'image': img_data, 'label': label_data}