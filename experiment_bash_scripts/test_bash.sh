
BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"


# todo itm: nnUNetTrainerConvexHullUMambaEnc__nnUNetPlans_conv_Enc_y__3d_fullres


OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetTrainerConvexHullUMambaEnc_y"
OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_pp/nnUNetTrainerConvexHullUMambaEnc_y"
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_conv_Enc_y --verify_dataset_integrity

mkdir -p $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER_PP

nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans_wo_Mamba
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerConvexHullUMambaEnc -p nnUNetPlans_conv_Enc_y -f 0 1 2 3

nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerConvexHullUMambaEnc__nnUNetPlans_conv_Enc_y__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerConvexHullUMambaEnc__nnUNetPlans_conv_Enc_y__3d_fullres/crossval_results_folds_0_1_2_3/plans.json


