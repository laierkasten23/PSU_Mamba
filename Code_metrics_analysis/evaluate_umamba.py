import os
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import gc


def extract_prefix(filename):
    base_name = os.path.basename(filename)
    if '_' in base_name:
        return base_name.split('_')[0]
    else:
        return base_name.split('.')[0]

    
# Function to determine model and modality from pred_folder
def get_model_modality(pred_folder):
    if 'aschoplex' in pred_folder:
        model = 'aschoplex'
    elif 'phusegplex' in pred_folder:
        model = 'phusegplex'
    elif 'umamba' in pred_folder:
        model = 'umamba'
    else:
        model = 'unknown'

    if 'T1/' in pred_folder:
        modality = 'T1'
    elif '_FLAIR/' in pred_folder:
        modality = 'FLAIR'
    elif 'T1xFLAIR/' in pred_folder:
        modality = 'T1xFLAIR'
    elif 'T1_FLAIR_T1xFLAIRmask/' in pred_folder:
        modality = 'T1_FLAIR_T1xFLAIRmask'
    else:
        modality = 'unknown'

    return f"{model}_{modality}"


def evaluate_all_experiments(pred_folders, gt_folder, save_csv_path):
    """
    Evaluate all experiments by computing various metrics for each prediction folder against the ground truth.
    Parameters:
    pred_folders (list of str): List of paths to the folders containing prediction files.
    gt_folder (str): Path to the folder containing ground truth files.
    save_csv_path (str): Path to save the results as a CSV file.
    Returns:
    None
    The function performs the following steps:
    1. Loads ground truth files from the specified folder.
    2. Initializes metrics for evaluation: Dice, Hausdorff Distance, and Confusion Matrix (Precision, Recall, F1 Score).
    3. Iterates over each prediction folder to:
        a. Load prediction files.
        b. Compute metrics for each pair of prediction and ground truth.
        c. Track best and worst segmentations based on Dice and Hausdorff Distance.
    4. Aggregates the metrics to get summary statistics for each prediction folder.
    5. Prints the results in a tabular format.
    6. Saves the results to a CSV file at the specified path.
    """
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.nii')])
    ground_truths = {extract_prefix(f): nib.load(os.path.join(gt_folder, f)).get_fdata() for f in gt_files}


    # Initialize the metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    confusion_matrix_metric = ConfusionMatrixMetric(include_background=False, metric_name=["precision", "recall", "f1_score"], reduction="mean")

    # List to store the results
    results_list = []

    # Loop over each prediction folder
    for pred_folder in pred_folders:
        # Get model and modality
        model_modality = get_model_modality(pred_folder)
        print(f"Evaluating {model_modality}...")
        pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')])
        #predictions = [nib.load(os.path.join(pred_folder, f)).get_fdata() for f in pred_files]
        dice_scores = []
        hd_distances = []
        f1_scores = []
        precisions = []
        recalls = []

        # Initialize variables to store best and worst segmentations
        best_dice_score = -1
        worst_dice_score = float('inf')
        best_hd_distance = float('inf')
        worst_hd_distance = -1

        best_dice_filename = None
        worst_dice_filename = None
        best_hd_filename = None
        worst_hd_filename = None
        #for pred, gt, filename in zip(predictions, ground_truths, pred_files):
        # Compute metrics for each pair of prediction and ground truth
        for pred_file in pred_files:
            pred_prefix = extract_prefix(pred_file)
            print("pred_file", pred_file)
            
            if pred_prefix not in ground_truths:
                print(f"Warning: No matching ground truth for prediction file {pred_file}")
                continue

            gt = ground_truths[pred_prefix]
            pred_path = os.path.join(pred_folder, pred_file)
            pred = nib.load(pred_path).get_fdata()
            pred_tensor = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Check for empty tensors
            if torch.sum(pred_tensor) == 0 or torch.sum(gt_tensor) == 0:
                print("Warning: Empty prediction or ground truth tensor detected.")
                continue
            
            # Ensure matching shapes
            if pred_tensor.shape != gt_tensor.shape:
                print(f"Error: Shape mismatch - pred: {pred_tensor.shape}, gt: {gt_tensor.shape}")
                continue
            
            # Compute Dice score
            dice_score = dice_metric(y_pred=pred_tensor, y=gt_tensor)
            mean_dice_score = dice_score.mean().item()
            dice_scores.append(np.round(mean_dice_score,4))

            # Update best and worst Dice segmentations
            if mean_dice_score > best_dice_score:
                best_dice_score = mean_dice_score
                best_dice_prefix = extract_prefix(pred_file)
            if mean_dice_score < worst_dice_score:
                worst_dice_score = mean_dice_score
                worst_dice_prefix = extract_prefix(pred_file)
            
            # Compute Hausdorff distance
            hd_distance = hausdorff_metric(y_pred=pred_tensor, y=gt_tensor)

            hd_distances.append(np.round(hd_distance.item(),4))

            # Update best and worst Hausdorff segmentations
            if hd_distance < best_hd_distance:
                best_hd_distance = hd_distance.item()
                best_hd_prefix = extract_prefix(pred_file)
            if hd_distance > worst_hd_distance:
                worst_hd_distance = hd_distance.item()
                worst_hd_prefix = extract_prefix(pred_file)
            
            # Accumulate confusion matrix results
            confusion_matrix_metric(y_pred=pred_tensor, y=gt_tensor)
            precision, recall, f1_score = confusion_matrix_metric.aggregate()
            
            f1_scores.append(np.round(f1_score.item(),4))
            precisions.append(np.round(precision.item(),4))
            recalls.append(np.round(recall.item(),4))

            # Free up memory
            del pred_tensor, gt_tensor
            gc.collect()

        # Aggregate metrics to get summary statistics
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_hd_distance = sum(hd_distances) / len(hd_distances)
        mean_precision = sum(precisions) / len(precisions)
        mean_recall = sum(recalls) / len(recalls)
        mean_f1_score = sum(f1_scores) / len(f1_scores)

        # Append the results to the list
        results_list.append({
            "Model": model_modality,
            "Mean Dice": np.round(mean_dice,4),
            "Mean HD": np.round(mean_hd_distance,4),
            "Precision": np.round(mean_precision,4),
            "Recall": np.round(mean_recall,4),
            "F1 Score": np.round(mean_f1_score,4),
            "Best Dice": best_dice_prefix,
            "Worst Dice": worst_dice_prefix,
            "Best HD": best_hd_prefix,
            "Worst HD": worst_hd_prefix
        })


    # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Print the results table
    print(results)

    # Save the DataFrame to a CSV file
    results.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--pred_folders", nargs='+', required=True, help="List of prediction folders.")
    parser.add_argument("--gt_folder", required=True, help="Folder containing ground truth files.")
    parser.add_argument("--save_csv_path", required=True, help="Path to save the results CSV file.")
    
    args = parser.parse_args()
    print("In evaluate_umamba.py")
    print(args)
    evaluate_all_experiments(args.pred_folders, args.gt_folder, args.save_csv_path)
    
    # python evaluate_umamba.py \
  # --pred_folders /data1/umamba_predictions/working_directory_T1xFLAIR/pred_raw/ \
  # --gt_folder /data1/Umamba_data/nnUNet_raw/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/labelsTs \
  # --save_csv_path /data1/Umamba_data/results_umamba.csv
  
  # /mnt/reference_labels/ref_labelTs
  
#python3 evaluate_umamba.py \
#    --pred_folders "/data1/umamba_predictions/working_directory_T1xFLAIR/pred_raw/" \
#    --gt_folder "/mnt/reference_labels/ref_labelTs" \
#    --save_csv_path "/data1/Umamba_data/results_umamba.csv"

#python3 evaluate_umamba.py \
#    --pred_folders "/data1/Umamba_data/umamba_predictions/working_directory_T1xFLAIR/pred_raw/" \
#    --gt_folder "/mnt/reference_labels/ref_labelTs" \
#    --save_csv_path "/data1/Umamba_data/results_umamba.csv"