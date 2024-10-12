#!/bin/bash


EXP_FOLDER="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments"
AP_FOLDER="01_aschoplex_from_scratch"
PHUSE_FOLDER="02_phusegplex"
UMAMBA_FOLDER="umamba_predictions"


# Define the prediction folders
PRED_FOLDERS=(
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_FLAIR/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1xFLAIR/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1_FLAIR_T1xFLAIRmask/ensemble_output/image_Ts"
    "$EXP_FOLDER/$PHUSE_FOLDER/working_directory_02_T1/ensemble_output/image_Ts"
    "$EXP_FOLDER/$PHUSE_FOLDER/working_directory_02_FLAIR/ensemble_output/image_Ts" 
    "$EXP_FOLDER/$PHUSE_FOLDER/working_directory_02_T1xFLAIR/ensemble_output/image_Ts" 
    "$EXP_FOLDER/$PHUSE_FOLDER/working_directory_02_T1_FLAIR_T1xFLAIRmask/ensemble_output/image_Ts"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_FLAIR/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1xFLAIR/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1_FLAIR_T1xFLAIRmask/pred_pp"
)



# Define the ground truth folder
GT_FOLDER="/home/linuxlia/Lia_Masterthesis/data/reference_labels_T1/ref_labelTs"
GT_FOLDER="/home/linuxlia/Lia_Masterthesis/data/reference_labels/ref_labelTs"
GT_FOLDER="/home/linuxlia/Lia_Masterthesis/data/reference_labels_FLAIR/ref_labelTs"

# Define the path to save the CSV file
SAVE_CSV_PATH="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/segmentation_metrics_t1_gt.csv"
SAVE_CSV_PATH="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/segmentation_metrics_t1xflair_gt.csv"
SAVE_CSV_PATH="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/segmentation_metrics_flair_gt.csv"

# Call the Python script with the arguments
python evaluate.py --pred_folders "${PRED_FOLDERS[@]}" --gt_folder "$GT_FOLDER" --save_csv_path "$SAVE_CSV_PATH"