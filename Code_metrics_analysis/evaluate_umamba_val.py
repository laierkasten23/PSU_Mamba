import os
import nibabel as nib
import torch
import pandas as pd
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
import argparse
import gc
import gzip
import tempfile
import shutil

def safe_nifti_load(path):
    # Read magic number
    with open(path, 'rb') as f:
        magic = f.read(2)

    try:
        if magic == b'\x1f\x8b':  # gzip magic number
            return nib.load(path)
        else:
            # Create a temporary .nii file with correct content
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
                shutil.copyfile(path, tmp_file.name)
                return nib.load(tmp_file.name)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        raise

def extract_prefix(filename):
    base_name = os.path.basename(filename)
    if '_' in base_name:
        return base_name.split('_')[0]
    else:
        return base_name.split('.')[0]

def evaluate_model_all_folds_together(model_folder, gt_folder):
    # Find all fold_x/validation folders
    mod_name = os.path.basename(model_folder.rstrip('/'))
    
    fold_dirs = [os.path.join(model_folder, d, "validation") for d in os.listdir(model_folder)
                 if d.startswith("fold_") and os.path.isdir(os.path.join(model_folder, d, "validation"))]
    if not fold_dirs:
        print(f"No validation folders found in {model_folder}")
        return None

    # Load ground truths
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    ground_truths = {}
    for f in gt_files:
        gt_path = os.path.join(gt_folder, f)
        try:
            gt_img = safe_nifti_load(gt_path)
            ground_truths[extract_prefix(f)] = gt_img.get_fdata()
        except Exception as e:
            print(f"Could not load {gt_path}: {e}")
    
    #print(f"Ground truth files found: {gt_files}")
    #print(f"Ground truths loaded: {list(ground_truths.keys())}")

    # Prepare metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    confusion_matrix_metric = ConfusionMatrixMetric(include_background=False, metric_name=["precision", "recall", "f1_score"], reduction="mean")

    # Results for all predictions (across all folds)
    dice_scores, hd_distances, f1_scores, precisions, recalls = [], [], [], [], []
    best_dice_score, worst_dice_score = -1, float('inf')
    best_hd_distance, worst_hd_distance = float('inf'), -1
    best_dice_filename = worst_dice_filename = best_hd_filename = worst_hd_filename = None

    for fold_dir in fold_dirs:
        pred_files = sorted([f for f in os.listdir(fold_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        for pred_file in pred_files:
            pred_prefix = extract_prefix(pred_file)
            # if prefix 015, 047, 80 or 098, skip
            if mod_name == 'nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_global_pca' and pred_prefix in ['057', '028', '043', '025', '020', '031', '021']: # ['015', '047', '080', '098'], ['057', '028'], ['098', '047'], ['015', '080']
                print(f"Skipping prediction file {pred_file} with prefix {pred_prefix}")
                continue
            
            if mod_name == 'nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_x' and pred_prefix in ['020', '028', '052']: 
                print(f"Skipping prediction file {pred_file} with prefix {pred_prefix}")
                continue
            
            if mod_name == 'nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_y' and pred_prefix in ['057', '052', '043', '025']: 
                print(f"Skipping prediction file {pred_file} with prefix {pred_prefix}")
                continue
            
            if pred_prefix not in ground_truths:
                print(f"Warning: No matching ground truth for prediction file {pred_file}")
                continue

            gt = ground_truths[pred_prefix]
            pred_path = os.path.join(fold_dir, pred_file)
            #print(f"Comparing prediction: {pred_path} with ground truth: {pred_prefix}")
            pred = nib.load(pred_path).get_fdata()
            pred_tensor = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Check for empty tensors
            if torch.sum(pred_tensor) == 0 or torch.sum(gt_tensor) == 0:
                print(f"Warning: Empty prediction or ground truth tensor for {pred_file}")
                continue

            # Ensure matching shapes
            if pred_tensor.shape != gt_tensor.shape:
                print(f"Error: Shape mismatch - pred: {pred_tensor.shape}, gt: {gt_tensor.shape}")
                continue

            # Dice
            dice_score = dice_metric(y_pred=pred_tensor, y=gt_tensor)
            mean_dice_score = dice_score.mean().item()
            dice_scores.append(mean_dice_score)

            if mean_dice_score > best_dice_score:
                best_dice_score = mean_dice_score
                best_dice_filename = pred_file
            if mean_dice_score < worst_dice_score:
                worst_dice_score = mean_dice_score
                worst_dice_filename = pred_file

            # Hausdorff
            hd_distance = hausdorff_metric(y_pred=pred_tensor, y=gt_tensor)
            hd_val = hd_distance.item()
            hd_distances.append(hd_val)

            if hd_val < best_hd_distance:
                best_hd_distance = hd_val
                best_hd_filename = pred_file
            if hd_val > worst_hd_distance:
                worst_hd_distance = hd_val
                worst_hd_filename = pred_file

            # Confusion matrix
            confusion_matrix_metric(y_pred=pred_tensor, y=gt_tensor)
            precision, recall, f1_score = confusion_matrix_metric.aggregate()
            print(f"Dice: {mean_dice_score:.4f}, HD: {hd_val:.4f}, Precision: {precision.item():.4f}, Recall: {recall.item():.4f}, F1 Score: {f1_score.item():.4f}")
            f1_scores.append(f1_score.item())
            precisions.append(precision.item())
            recalls.append(recall.item())

            del pred_tensor, gt_tensor
            gc.collect()

    if dice_scores:
        return {
            "Model": mod_name,
            "Mean Dice": np.round(np.mean(dice_scores), 4),
            "Std Dice": np.round(np.std(dice_scores), 4),
            "Mean HD": np.round(np.mean(hd_distances), 4),
            "Std HD": np.round(np.std(hd_distances), 4),
            "Mean Precision": np.round(np.mean(precisions), 4),
            "Std Precision": np.round(np.std(precisions), 4),
            "Mean Recall": np.round(np.mean(recalls), 4),
            "Std Recall": np.round(np.std(recalls), 4),
            "Mean F1": np.round(np.mean(f1_scores), 4),
            "Std F1": np.round(np.std(f1_scores), 4),
            "Best Dice": np.round(best_dice_score, 4),
            "Best Dice File": best_dice_filename,
            "Worst Dice": np.round(worst_dice_score, 4),
            "Worst Dice File": worst_dice_filename,
            "Best HD": np.round(best_hd_distance, 4),
            "Best HD File": best_hd_filename,
            "Worst HD": np.round(worst_hd_distance, 4),
            "Worst HD File": worst_hd_filename,
        }
    else:
        print(f"No valid predictions found in {model_folder}")
        return None

def evaluate_all_models(parent_folder, gt_folder, save_csv_path, model_names=None, csv_name="all_models_validation_metrics.csv"):
    if model_names is not None:
        model_folders = [os.path.join(parent_folder, d) for d in model_names if os.path.isdir(os.path.join(parent_folder, d))]
    else:
        model_folders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
                         if os.path.isdir(os.path.join(parent_folder, d))]
    all_results = []
    csv_path = os.path.join(save_csv_path, csv_name)
    for model_folder in model_folders:
        print(f"Evaluating model: {model_folder}")
        result = evaluate_model_all_folds_together(model_folder, gt_folder)
        if result is not None:
            all_results.append(result)
            # Write to CSV after every model
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"Saved current results to {csv_path}")
    print(f"Saved all model results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all model folds in a parent folder (aggregated).")
    parser.add_argument("--models_folder", required=True, help="Folder containing model subfolders.")
    parser.add_argument("--gt_folder", required=True, help="Folder containing ground truth files.")
    parser.add_argument("--save_csv_path", required=True, help="Folder to save the results CSV file.")
    parser.add_argument("--model_names", nargs='*', default=None, help="Optional list of model folder names to validate (space-separated).")
    args = parser.parse_args()

    os.makedirs(args.save_csv_path, exist_ok=True)
    evaluate_all_models(args.models_folder, args.gt_folder, args.save_csv_path, model_names=args.model_names)
    
 # 
    # Example usage:
    """
    python evaluate_umamba_val.py --models_folder /data1/user/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA \
        --gt_folder /data1/user/Umamba_data/nnUNet_raw/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/labelsTr \
        --save_csv_path /data1/user/Umamba_data/results_umamba_val_Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA \
        --model_names nnUNetTrainerUMambaBot__nnUNetPlans_First_general__3d_fullres nnUNetTrainerUMambaEnc__nnUNetPlans_First_general__3d_fullres nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general__3d_fullres_x nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general__3d_fullres_y nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general__3d_fullres_z nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general__3d_fullres_global_pca nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_64__3d_fullres_x nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_64__3d_fullres_y nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_64__3d_fullres_z nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_64__3d_fullres_xy-diag nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_64__3d_fullres_global_pca nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_32__3d_fullres_global_pca
    
    python evaluate_umamba_val.py --models_folder /data1/user/Umamba_data/nnUNet_results/Dataset299_BrainStem \
        --gt_folder /data1/user/Umamba_data/nnUNet_raw/Dataset299_BrainStem/labelsTr \
        --save_csv_path /data1/user/Umamba_data/results_umamba_val_Dataset299_BrainStem \
        --model_names nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_global_pca nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_x nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba__nnUNetPlans_First_general_128__3d_fullres_y
    """
    # 
    
# This script evaluates segmentation predictions across all folds of a model, aggregating results into a single CSV file.