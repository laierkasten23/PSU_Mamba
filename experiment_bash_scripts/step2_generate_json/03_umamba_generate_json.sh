#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"

mode=train_predict
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasettype=UMAMBA




python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset033_ChoroidPlexus_T1_sym_UMAMBA" \
--datasettype $datasettype \
--modality "['T1']"  
        
python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset033_ChoroidPlexus_FLAIR_sym_UMAMBA" \
--datasettype $datasettype \
--modality "['FLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset033_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" \
--datasettype $datasettype \
--modality "['T1xFLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset033_ChoroidPlexus_T1_FLAIR_sym_UMAMBA" \
--datasettype $datasettype \
--modality "['T1', 'FLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset033_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA" \
--datasettype $datasettype \
--modality "['T1', 'FLAIR']" 
