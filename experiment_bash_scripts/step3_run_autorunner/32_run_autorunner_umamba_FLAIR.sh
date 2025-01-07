# Activate the Conda environment
source /home/linuxlia/miniconda3/bin/activate umamba

# Print the Python version
python --version

python import nibabel

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"
OUTPUT_FOLDER="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_FLAIR/pred_raw"
OUTPUT_FOLDER_PP="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_FLAIR/pred_pp"

mkdir -p "$OUTPUT_FOLDER"

# Define the path to the Conda environment's bin directory
CONDA_BIN_PATH=/home/linuxlia/miniconda3/envs/umamba/bin


# Add the directory containing the custom trainer class to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/03_U-Mamba/umamba/nnunetv2


# Preprocessing
nnUNetv2_plan_and_preprocess -d 432 --verify_dataset_integrity


nnUNetv2_train 432 3d_fullres 0 -tr nnUNetTrainerUMambaBot --npz
nnUNetv2_train 432 3d_fullres 1 -tr nnUNetTrainerUMambaBot --npz
nnUNetv2_train 432 3d_fullres 2 -tr nnUNetTrainerUMambaBot --npz
nnUNetv2_train 432 3d_fullres 3 -tr nnUNetTrainerUMambaBot --npz



# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot
nnUNetv2_find_best_configuration 432 -c 3d_fullres -tr nnUNetTrainerUMambaBot -f 0 1 2 3

***Run inference like this:***

nnUNetv2_predict -d Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaBot -c 3d_fullres -p nnUNetPlans

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results/Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results/Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3/plans.json

# Inference
#nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 332 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta