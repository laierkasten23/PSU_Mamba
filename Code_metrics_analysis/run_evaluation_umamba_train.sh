#!/bin/bash


EXP_FOLDER="/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results"
UMAMBA_BOT_FOLDER="nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres"

# Define the prediction folders

PRED_FOLDERS=(
    "$EXP_FOLDER/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/$UMAMBA_BOT_FOLDER/fold_0/validation"
    "$EXP_FOLDER/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/$UMAMBA_BOT_FOLDER/fold_1/validation"
    "$EXP_FOLDER/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/$UMAMBA_BOT_FOLDER/fold_2/validation"
    "$EXP_FOLDER/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/$UMAMBA_BOT_FOLDER/fold_3/validation"
)

# Define the ground truth folder
GT_FOLDER="/home/linuxlia/Lia_Masterthesis/data/reference_labels_T1/ref_labelTr"

# Define the path to save the CSV file
SAVE_CSV_PATH="/home/linuxlia/Lia_Masterthesis/data/Umamba_data/segmentation_metrics_ex_1.csv"

# Call the Python script with the arguments
python evaluate_umamba.py --pred_folders "${PRED_FOLDERS[@]}" --gt_folder "$GT_FOLDER" --save_csv_path "$SAVE_CSV_PATH"