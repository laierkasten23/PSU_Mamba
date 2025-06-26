#!/bin/bash

# Define the base directory for the project
# BASE_DIR="/home/linuxlia/Lia_Masterthesis"
# BASE_DIR="/home/studenti/facchi/lia_masterthesis"
BASE_DIR="/home/studenti/lia/lia_masterthesis"
BASE_DIR="/home/studenti/lia/projects"

mode=train_predict
# BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
BASE_DATA_DIR="/data1/LIA/Umamba_data/nnUNet_raw"
BASE_DATA_DIR="/data1/LIA/Umamba_data/nnUNet_raw"
datasettype=UMAMBA
fileending=".nii.gz"
# benchmark_dataroot="$BASE_DIR/data/reference_labels"
benchmark_dataroot="/var/datasets/LIA/reference_labels"
benchmark_dataroot="/data1/LIA/reference_labels"
# groups='/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
# groups='/var/datasets/LIA/pazienti/patients.json'
groups='/mnt/turing/LIA/pazienti/patients.json'
groups='/mnt/LIA/pazienti/patients.json'



#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset431_ChoroidPlexus_T1_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1']"  
        
#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['FLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" \
--datasettype $datasettype \
--fileending $fileending \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--groups $groups \
--modality "['T1xFLAIR']" 

#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset434_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1', 'FLAIR']" 

#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset435_ChoroidPlexus_T1_FLAIR_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1', 'FLAIR']" 


#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset994_Dir_z" \
#--datasettype $datasettype \
#--fileending $fileending \
#--modality "['T1xFLAIR']" 
