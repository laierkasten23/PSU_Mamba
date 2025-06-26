import os
import nibabel as nib
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageViewer:
    def __init__(self, dataroot, modality, ensemble_folder):
        self.dataroot = dataroot
        self.modality = modality
        self.ensemble_folder = ensemble_folder
        self.ensemble_files = [f for f in os.listdir(ensemble_folder) if f.endswith('.nii.gz')]
        self.current_index = 0
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.next_button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.next_button.on_clicked(self.next_image)
        print(f"Found {len(self.ensemble_files)} ensemble files")
        self.show_image()

    def show_image(self):
        if self.current_index >= len(self.ensemble_files):
            plt.close(self.fig)
            return
        
        ensemble_file = self.ensemble_files[self.current_index]
        identifier = ensemble_file.split('_')[0]
        subject_folder = os.path.join(self.dataroot, identifier)
        modality_file = os.path.join(subject_folder, f"{identifier}_{self.modality}.nii")
        true_mask_file = os.path.join(subject_folder, f"{identifier}_ChP_mask_{self.modality}_manual_seg.nii")
        ensemble_file_path = os.path.join(self.ensemble_folder, ensemble_file)
        
        modality_img = nib.load(modality_file).get_fdata()
        true_mask_img = nib.load(true_mask_file).get_fdata()
        ensemble_img = nib.load(ensemble_file_path).get_fdata()
        
        self.display_3d_plot(modality_img, true_mask_img, ensemble_img)
        
    def display_3d_plot(self, modality_img, true_mask_img, ensemble_img):
        fig = mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
        mlab.contour3d(modality_img, contours=10, opacity=0.1, colormap='gray')
        mlab.contour3d(true_mask_img, contours=[0.5], opacity=0.5, color=(1, 0, 0))
        mlab.contour3d(ensemble_img, contours=[0.5], opacity=0.5, color=(0, 1, 0))
        
        self.rotate_plot(fig)
        mlab.show()

    def rotate_plot(self, fig):
        @mlab.animate(delay=100)
        def anim():
            while True:
                for angle in range(0, 360, 10):
                    mlab.view(azimuth=angle)
                    yield
        anim()

    def next_image(self, event):
        self.current_index += 1
        self.show_image()
        print(f"Showing image {self.current_index + 1} of {len(self.ensemble_files)}")


# Example usage
dataroot = "/home/linuxuser/user/data/pazienti"
modality = "T1"
ensemble_folder = "/home/linuxuser/user/project_dir/_experiments/01_aschoplex_from_scratch/working_directory_T1_240820_1823/ensemble_output/image_Ts/"
viewer = ImageViewer(dataroot, modality, ensemble_folder)
print('Viewer created')
plt.show()