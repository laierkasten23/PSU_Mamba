import argparse
import os
import numpy as np
from scipy import stats
import nibabel as nib

from scipy.spatial.distance import dice
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score, cohen_kappa_score
from scipy.stats import wilcoxon, ttest_rel

def load_nifti_file(filepath):
    """ Load a NIfTI file and return the image data as a numpy array. """
    img = nib.load(filepath)
    return img.get_fdata()

def calculate_volume(nifti_file):
    """
    Calculate the volume of non-zero values in a NIfTI file.

    Parameters:
    nifti_file (str): The path to the NIfTI file.

    Returns:
    int: The volume of non-zero values in the NIfTI file.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    volume = np.count_nonzero(data)
    return volume

def calculate_dice(nifti_file1, nifti_file2):
    """
    Calculates the Dice coefficient between two NIfTI files.

    Parameters:
    nifti_file1 (str): Path to the first NIfTI file.
    nifti_file2 (str): Path to the second NIfTI file.

    Returns:
    float: The Dice coefficient between the two NIfTI files.
    """
    img1 = nib.load(nifti_file1)
    data1 = img1.get_fdata()

    img2 = nib.load(nifti_file2)
    data2 = img2.get_fdata()

    intersection = np.sum(data1 * data2)
    dice = (2. * intersection) / (np.sum(data1) + np.sum(data2))
    return dice


def dice_coefficient(mask1, mask2):
    """ Compute the Dice Similarity Coefficient for two binary masks. """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    if size1 + size2 == 0:
        return 1.0  # Both masks are empty, return perfect similarity.
    return 2.0 * intersection / (size1 + size2)





def jaccard_index(mask1, mask2):
    """ Compute the Jaccard Index for two binary masks. """
    mask1 = mask1.astype(bool).flatten()
    mask2 = mask2.astype(bool).flatten()
    return jaccard_score(mask1, mask2)

def cohen_kappa(mask1, mask2):
    """ Compute Cohen's Kappa for two binary masks. """
    mask1 = mask1.astype(bool).flatten()
    mask2 = mask2.astype(bool).flatten()
    return cohen_kappa_score(mask1, mask2)

def hausdorff_dist(mask1, mask2):
    """ Compute the Hausdorff Distance for two binary masks. """
    return hausdorff_distance(mask1, mask2)

def volume(mask):
    """ Compute the volume (number of voxels) for a binary mask. """
    return int(np.sum(mask))


def process_subjects(subjects_dir, file_ending1 = '_ChP_mask_T1_mdc.nii', file_ending2 = '_ChP_mask_T1xFLAIR.nii'):
    """ Process NIfTI files for all subjects and compute statistical measures for each pair of masks. """
    subjects = [f.name for f in os.scandir(subjects_dir) if f.is_dir()]

    dsc_scores, jaccard_scores, kappa_scores, hausdorff_scores, volumes_t1c, volumes_flair = [], [], [], [], [], []

    for subject in subjects:
        t1mc_path = os.path.join(subjects_dir, subject, subject + file_ending1)
        t1xflair_path = os.path.join(subjects_dir, subject, subject + file_ending2)
        
        if not os.path.exists(t1mc_path) or not os.path.exists(t1xflair_path):
            print(f"Skipping subject {subject}: Missing data")
            continue
        
        t1mc_mask = load_nifti_file(t1mc_path)
        t1xflair_mask = load_nifti_file(t1xflair_path)
        
        dsc = dice_coefficient(t1mc_mask, t1xflair_mask)
        jaccard = jaccard_index(t1mc_mask, t1xflair_mask)
        kappa = cohen_kappa(t1mc_mask, t1xflair_mask)
        hausdorff = hausdorff_dist(t1mc_mask, t1xflair_mask)
        vol_t1mc = volume(t1mc_mask)
        vol_t1xflair = volume(t1xflair_mask)
        
        dsc_scores.append(dsc)
        jaccard_scores.append(jaccard)
        kappa_scores.append(kappa)
        hausdorff_scores.append(hausdorff)
        volumes_t1c.append(vol_t1mc)
        volumes_flair.append(vol_t1xflair)
        
        print(f"Subject {subject}: DSC = {dsc:.4f}, Jaccard = {jaccard:.4f}, Kappa = {kappa:.4f}, Hausdorff = {hausdorff:.4f}, Volume T1c = {vol_t1mc}, Volume FLAIR = {vol_t1xflair}")

    return dsc_scores, jaccard_scores, kappa_scores, hausdorff_scores, volumes_t1c, volumes_flair

argparser = argparse.ArgumentParser(description="Compute statistical measures for T1c and T1xFLAIR masks.")
argparser.add_argument("subjects_dir", type=str, help="Path to the directory containing the subject data.")
args = argparser.parse_args()

# Main

if __name__ == "__main__":

    # Define the path to the directory containing the subject data
    print("T1mc vs T1xFLAIR")
    subjects_dir = '/home/linuxlia/Lia_Masterthesis/data/T1mc_vs_T1xFLAIR_controlled_OK'

    # Process the subjects and compute statistical measures
    dsc_scores, jaccard_scores, kappa_scores, hausdorff_scores, volumes_t1mc, volumes_t1xflair = process_subjects(subjects_dir)

    # Perform statistical tests
    # Wilcoxon signed-rank test
    wilcoxon_volumes = wilcoxon(volumes_t1mc, volumes_t1xflair)

    # Paired t-test
    ttest_volumes = ttest_rel(volumes_t1mc, volumes_t1xflair)

    print("\nWilcoxon signed-rank test results:")
    print(f"Volumes: statistic={wilcoxon_volumes.statistic}, p-value={wilcoxon_volumes.pvalue}")

    print("\nPaired t-test result:")
    print(f"Volumes T1c vs T1xFLAIR: statistic={ttest_volumes.statistic}, p-value={ttest_volumes.pvalue}")
