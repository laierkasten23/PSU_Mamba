import matplotlib.pyplot as plt
import os

def load_png_image(filepath):
    """Load a png image and return the data array."""
    return plt.imread(filepath)

def plot_slices(subject_ids, base_dir):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # Adjust figsize if necessary
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.02, wspace=0.02, hspace=0.02)

    
    row_labels = ['3D T1xFLAIR', 'T1 axial', 'FLAIR coronal']
    col_labels = ['SMC', 'MCI', 'AD', 'Other']

    for col, subject_id in enumerate(subject_ids):
        # Load images
        t1xflair_vol_path = os.path.join(base_dir, f'{subject_id}_T1xFLAIR_vol.png') 
        t1_img_path = os.path.join(base_dir, f'{subject_id}_T1_axial.png')
        flair_img_path = os.path.join(base_dir, f'{subject_id}_FLAIR_coronal.png')

        # Load data
        t1xflair_vol = load_png_image(t1xflair_vol_path)
        t1_img = load_png_image(t1_img_path)
        flair_img = load_png_image(flair_img_path)

        # First row: T1xFLAIR
        axes[0, col].imshow(t1xflair_vol)
        axes[0, col].set_title(col_labels[col], fontsize=16)
        #axes[0, col].axis('on')

        # Second row: T1 with segmentation
        axes[1, col].imshow(t1_img)
        #axes[1, col].axis('off')

        # Third row: FLAIR with segmentation
        axes[2, col].imshow(flair_img)
        #axes[2, col].axis('off')

    ## Set row labels for the first column
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(row_labels[i], rotation=90, fontsize=16, labelpad=20, va='center')

    # Hide ticks and spines for all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')

    # Save plot as png
    plt.savefig(os.path.join(base_dir, 'group_all_modalities_plot.png'), dpi=300)
    print("Saved to", os.path.join(base_dir, 'group_all_modalities_plot.png'))

    plt.show()

# Example usage
subject_ids = ['025', '088', '017', '021']
base_dir = '/home/linuxlia/Lia_Masterthesis/data/thesis_images'
plot_slices(subject_ids, base_dir)
