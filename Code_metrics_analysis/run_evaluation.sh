#!/bin/bash


EXP_FOLDER="/home/linuxuser/user/project_dir/_experiments"
EXP_FOLDER="/data1/user/Umamba_data"
AP_FOLDER="01_aschoplex_from_scratch"
lab_FOLDER="02_labgplex"
UMAMBA_FOLDER="umamba_predictions"
um_loc="working_directory_T1xFLAIR/pred_raw"


# Define the prediction folders
PRED_FOLDERS=(
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_FLAIR/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1xFLAIR/ensemble_output/image_Ts"
    "$EXP_FOLDER/$AP_FOLDER/working_directory_01_T1_FLAIR_T1xFLAIRmask/ensemble_output/image_Ts"
    "$EXP_FOLDER/$lab_FOLDER/working_directory_02_T1/ensemble_output/image_Ts"
    "$EXP_FOLDER/$lab_FOLDER/working_directory_02_FLAIR/ensemble_output/image_Ts" 
    "$EXP_FOLDER/$lab_FOLDER/working_directory_02_T1xFLAIR/ensemble_output/image_Ts" 
    "$EXP_FOLDER/$lab_FOLDER/working_directory_02_T1_FLAIR_T1xFLAIRmask/ensemble_output/image_Ts"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_FLAIR/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1xFLAIR/pred_pp"
    "$EXP_FOLDER/$UMAMBA_FOLDER/working_directory_T1_FLAIR_T1xFLAIRmask/pred_pp"
)

PRED_FOLDERS=( #nnUNetTrainerMambaFirstStem_diag
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetPlans_Mamba_1st"  # 0.8673   1.5602     0.8714  0.8651     0.868   091.nii    029.nii  004.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetPlans_Mamba_1st_stem_PCA" #     0.8671   1.6309     0.8627   0.864    0.8631   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirst_diag"   # 0.8654   1.5794     0.8554  0.8707    0.8628   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_diag"   # 0.45   19.692     0.6775  0.3552    0.4605   032.nii    027.nii  104.nii  027.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_PCA_32_with_patch_size_8"  # 0.8682   1.6489     0.8553   0.874    0.8642   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_x" #!  # Warning: Empty prediction or ground truth tensor detected.
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_z" # 0.8665,0.0459,1.5694,1.8658,0.8625,0.0204,0.8678,0.0114,0.8649,0.008,091.nii,029.nii,005.nii,029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerPCApath_First_PCA" # 0.8357   1.8889     0.8732    0.81    0.8402   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirst_diag"   # 0.8654   1.5794     0.8554  0.8707    0.8628   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_diag" #- >empty #!
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirstStem_PCA_patch16_32"    NO SUCH DIR
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerPCApath_First_PCA" # not in stem -> exclude 0.8357   1.8889     0.8732    0.81    0.8402   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerPCApath_First_PCA_raw"     # 0.8379   1.9506     0.8665   0.818    0.8413   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerPCApath_First_PCA_resnet"  # 0.85   1.7515     0.8647   0.838    0.8509   091.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerUMambaFirst_PCA_patch16_32"    # 0.8564   2.1927     0.8407  0.8754    0.8574   034.nii    029.nii  005.nii  029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerUMambaFirst_diag" # check ZeroDivisionError: division by zero
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerUMambaFirst_PCA_patch16_32" #  check
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaEnc_PRL" # 0.8635,0.0445,1.6702,1.8136,0.8657,0.0205,0.856,0.0114,0.8606,0.0101,091.nii,029.nii,005.nii,029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaBot_PRL" # 0.8696,0.0425,1.5537,1.8366,0.868,0.0211,0.869,0.0105,0.8683,0.0079,091.nii,029.nii,004.nii,029.nii
    #"$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirst_globalPCA_64_PRL" # 0.8811,0.0327,1.1723,0.3547,0.8712,0.0222,0.8846,0.0139,0.8775,0.0072,091.nii,055.nii,004.nii,035.nii
    "$EXP_FOLDER/$UMAMBA_FOLDER/$um_loc/nnUNetTrainerMambaFirst_globalPCA_32_PRL" # 0.8756,0.033,1.2831,0.5831,0.8646,0.0211,0.8799,0.014,0.8718,0.0069,091.nii,055.nii,004.nii,016.nii
)


# Define the ground truth folder
GT_FOLDER="/home/linuxuser/user/data/reference_labels_T1/ref_labelTs"
GT_FOLDER="/home/linuxuser/user/data/reference_labels/ref_labelTs"
GT_FOLDER="/home/linuxuser/user/data/reference_labels_FLAIR/ref_labelTs"
GT_FOLDER="/mnt/user/reference_labels/ref_labelTs"

# Define the path to save the CSV file
SAVE_CSV_PATH="/home/linuxuser/user/project_dir/_experiments/segmentation_metrics_t1_gt.csv"
SAVE_CSV_PATH="/home/linuxuser/user/project_dir/_experiments/segmentation_metrics_t1xflair_gt.csv"
SAVE_CSV_PATH="/home/linuxuser/user/project_dir/_experiments/segmentation_metrics_flair_gt.csv"
SAVE_CSV_PATH="/data1/user/Umamba_data/segmentation_metrics_t1xflair_gt_1.csv"
# Call the Python script with the arguments
python evaluate.py --pred_folders "${PRED_FOLDERS[@]}" --gt_folder "$GT_FOLDER" --save_csv_path "$SAVE_CSV_PATH"