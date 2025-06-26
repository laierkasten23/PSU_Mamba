# Define the base directory for the project
BASE_DIR="/home/linuxuser/user"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset434_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"
OUTPUT_FOLDER="/home/linuxuser/user/project_dir/_experiments/umamba_predictions/working_directory_T1_FLAIR_T1xFLAIRmask/pred_raw"
OUTPUT_FOLDER_PP="/home/linuxuser/user/project_dir/_experiments/umamba_predictions/working_directory_T1_FLAIR_T1xFLAIRmask/pred_pp"


nnUNetv2_plan_and_preprocess -d 434 --verify_dataset_integrity

cp /home/linuxuser/user/data/Umamba_data/nnUNet_raw/Dataset434_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA/splits_final.json /home/linuxuser/user/data/Umamba_data/nnUNet_preprocessed/Dataset434_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA/splits_final.json

nnUNetv2_plan_and_preprocess -d 434 -c 3d_fullres --verify_dataset_integrity --verbose


nnUNetv2_train 434 3d_fullres 0 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 434 3d_fullres 1 -tr nnUNetTrainerUMambaBot
#nnUNetv2_train 434 3d_fullres 2 -tr nnUNetTrainerUMambaBot
#nnUNetv2_train 434 3d_fullres 3 -tr nnUNetTrainerUMambaBot

#nnUNetv2_find_best_configuration 434 -c 3d_fullres -tr nnUNetTrainerUMambaBot -f 0 1 2 3