#!/bin/bash


# Define the prediction folders
PRED_FOLDERS=(
    "/home/linuxuser/user/data/ensemble_prediction_test_data_as_in__exp"
)


# Define the ground truth folder
GT_FOLDER="/home/linuxuser/user/data/reference_labels_T1/ref_labelTs"
GT_FOLDER="/home/linuxuser/user/data/reference_labels/ref_labelTs"
#GT_FOLDER="/home/linuxuser/user/data/reference_labels_FLAIR/ref_labelTs"

# Define the path to save the CSV file
SAVE_CSV_PATH="/home/linuxuser/user/project_dir/_experiments/segmentation_metrics_t1data_ap_finetuned.csv"


# Call the Python script with the arguments
python evaluate.py --pred_folders "${PRED_FOLDERS[@]}" --gt_folder "$GT_FOLDER" --save_csv_path "$SAVE_CSV_PATH"