import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


# Datasetclass for MRI niftii data

class MRI2D_z_data(Dataset):
    def __init__(self, train_test_path, transform=None, target_transform=None):
        """
        train_test_path -- path to data folder
        transform -- transform (from torchvision.transforms) to be applied to the data
        """
        self.path = train_test_path
        self.patients = [file for file in sorted(os.listdir(self.path)) if file not in ['.DS_Store','README.md']]
        self.samples = self._generate_samples()
        self.transform = transform # TODO
        self.target_transform = target_transform # TODO

    def _generate_samples(self):
        samples = []
        for subject_folder in self.patients:
            # Read data and mask
            data_file = os.path.join(self.path, subject_folder, 'T1.nii')  
            mask_file = os.path.join(self.path, subject_folder, 'mask.nii')  
            data = nib.load(data_file).get_fdata()
            mask = nib.load(mask_file).get_fdata()

            for z_slice in range(data.shape[2]):
                sample = {
                    'data': data[:, :, z_slice],
                    'mask': mask[:, :, z_slice],
                    'subject_idx': int(subject_folder)
                }
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        data_slice = np.expand_dims(sample['data'], axis=0) # Maybe remove, but this might be very usefull when stacking for the batch. 
        mask_slice = np.expand_dims(sample['mask'], axis=0)

        data_slice = torch.FloatTensor(data_slice)
        mask_slice = torch.FloatTensor(mask_slice)

        subject_idx_tensor = torch.tensor(sample['subject_idx'])

        # Apply Data Augmentation here if needed (TODO)
        if self.transform is not None:
            data_slice = self.transform(data_slice)
        if self.target_transform is not None:
            mask_slice = self.target_transform(mask_slice)

        return data_slice, mask_slice, subject_idx_tensor