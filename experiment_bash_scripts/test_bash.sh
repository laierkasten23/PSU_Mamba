#BASE_DATA_DIR="/data1/LIA/Umamba_data/nnUNet_raw"
#datasetname="Dataset994_Dir_z" 
#INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTr"

#OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/PRL/path_analysis_results/pred_raw"
#OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/PRL/path_analysis_results/pred_pp/"



nnUNetv2_train 433 3d_fullres 2 -tr nnUNetTrainerMambaFirstStem_PCA_2components_general -p nnUNetPlans_First_general --npz
#nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerMambaFirstStem_PCA_2components_general -p nnUNetPlans_First_general --npz --c




#BASE_DATA_DIR="/data1/LIA/Umamba_data/nnUNet_raw"
#datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
#INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"

#OUTPUT_FOLDER="/data1/LIA/Umamba_data/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetPlans_Mamba_1st_stem_PCA_32"
#OUTPUT_FOLDER_PP="/data1/LIA/Umamba_data/umamba_predictions/working_directory_T1xFLAIR//pred_pp/nnUNetPlans_Mamba_1st_stem_PCA_32"

#nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerMambaFirstStem_PCA -c 3d_fullres -p nnUNetPlans_First_PCA_32 --use_pca
#n ***Once inference is completed, run postprocessing like this:***

#nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /data1/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerMambaFirstStem_PCA__nnUNetPlans_First_PCA_32__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /data1/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerMambaFirstStem_PCA__nnUNetPlans_First_PCA_32__3d_fullres/crossval_results_folds_0_1_2_3/plans.json
