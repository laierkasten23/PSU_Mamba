import os
import numpy as np
import nibabel as nib
import argparse
import baycomp

def calculate_dice(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    volume_sum = np.sum(mask1 > 0) + np.sum(mask2 > 0)
    dice_coefficient = (2. * intersection) / volume_sum if volume_sum != 0 else 1.0
    return dice_coefficient

def compare_masks(t1xflair_path, t1mdc_path):
    # Load the masks
    t1xflair_mask = nib.load(t1xflair_path).get_fdata()
    t1mdc_mask = nib.load(t1mdc_path).get_fdata()

    # Ensure the masks have the same shape
    if t1xflair_mask.shape != t1mdc_mask.shape:
        raise ValueError(f"Shape mismatch: {t1xflair_path} and {t1mdc_path}")

    # Calculate Dice coefficient
    dice_coefficient = calculate_dice(t1xflair_mask, t1mdc_mask)
    return dice_coefficient

def process_folder(input_folder):
    dice_scores_t1xflair = []
    dice_scores_t1mdc = []

    for subfolder in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            t1xflair_path = os.path.join(subfolder_path, 'T1xFLAIR_mask.nii.gz')
            t1mdc_path = os.path.join(subfolder_path, 'T1mdc_mask.nii.gz')

            if os.path.exists(t1xflair_path) and os.path.exists(t1mdc_path):
                try:
                    dice_coefficient = compare_masks(t1xflair_path, t1mdc_path)
                    dice_scores_t1xflair.append(dice_coefficient)
                    dice_scores_t1mdc.append(dice_coefficient)
                except Exception as e:
                    print(f"Error processing {subfolder}: {e}")
            else:
                print(f"Missing masks in {subfolder}")

    return np.array(dice_scores_t1xflair), np.array(dice_scores_t1mdc)

def perform_bayesian_analysis(scores_t1xflair, scores_t1mdc):
    # Define ROPE (Region of Practical Equivalence)
    rope_t1xflair = 0.025  # Narrow around high dice
    rope_t1mdc = 0.05      # Slightly wider

    names_t1xflair = ("T1xFLAIR", "T1-CE")
    names_t1mdc = ("T1mdc", "T1-CE")
    names_t1xflair_vs_t1mdc = ("T1xFLAIR", "T1mdc")

    # Bayesian analysis
    prob_t1xflair = baycomp.two_on_single(scores_t1xflair, np.ones(len(scores_t1xflair)), rope=rope_t1xflair, plot=True, names=names_t1xflair)
    prob_t1mdc = baycomp.two_on_single(scores_t1mdc, np.ones(len(scores_t1mdc)), rope=rope_t1mdc, plot=True, names=names_t1mdc)
    prob_t1xflair_vs_t1mdc = baycomp.two_on_single(scores_t1xflair, scores_t1mdc, rope=rope_t1mdc, plot=True, names=names_t1xflair_vs_t1mdc)

    # Results
    print(f"T1xFLAIR vs T1-CE Probability within ROPE: {prob_t1xflair}")
    print(f"T1mdc vs T1-CE Probability within ROPE: {prob_t1mdc}")
    print(f"T1xFLAIR vs T1mdc Probability within ROPE: {prob_t1xflair_vs_t1mdc}")

def main():
    parser = argparse.ArgumentParser(description="Compare T1xFLAIR and T1mdc masks in subfolders and perform Bayesian analysis.")
    parser.add_argument('input_folder', type=str, help="Path to the input folder containing subfolders with masks.")
    args = parser.parse_args()

    scores_t1xflair, scores_t1mdc = process_folder(args.input_folder)
    perform_bayesian_analysis(scores_t1xflair, scores_t1mdc)

if __name__ == "__main__":
    main()