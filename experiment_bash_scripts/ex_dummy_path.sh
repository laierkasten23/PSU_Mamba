#BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
#datasetname="Dataset994_Dir_z" 
#INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTr"

#OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/PRL/path_analysis_results/pred_raw"
#OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/PRL/path_analysis_results/pred_pp/"

#nnUNetv2_plan_and_preprocess -d 994 -c 3d_fullres --verify_dataset_integrity
#nnUNetv2_plan_and_preprocess -d 994 -c 3d_fullres --verify_dataset_integrity

#nnUNetv2_train 994 3d_fullres 0 -tr nnUNetTrainerUMambaBot -p nnUNetPlans --npz


#nnUNetv2_train 994 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --npz

nnUNetv2_train 433 3d_fullres 1 -tr nnUNetTrainerPCApath -p nnUNetPlans_First_PCA --npz --c
nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerPCApath -p nnUNetPlans_First_PCA --npz --c
nnUNetv2_train 433 3d_fullres 2 -tr nnUNetTrainerPCApath -p nnUNetPlans_First_PCA --npz --c
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerPCApath -p nnUNetPlans_First_PCA --npz --c

BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"

OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetTrainerUMambaEnc_adaptive"
OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_pp/nnUNetTrainerUMambaEnc_adaptive"
#nnUNetTrainerUMambaEnc__nnUNetPlans_francesco
#nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr  nnUNetTrainerUMambaEnc -p nnUNetPlans_francesco -f 0

#nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaFirst -c 3d_fullres -p nnUNetPlans_First_diag
#nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaFirst__nnUNetPlans_First_diag__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaFirst__nnUNetPlans_First_diag__3d_fullres/crossval_results_folds_0_1_2_3/plans.json