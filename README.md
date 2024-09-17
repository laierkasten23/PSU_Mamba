# Phuse Thesis 2024

## General Guide
### Step 1:  Anonymize Data

This step is only necessary if the data you received is NOT anonymized.

Initial folder structure: 

```
python3 anonymize_dataset.py --path '/Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/phuse_thesis_2024/data/processed_data' --new_folder_name 'ANON_FLAIR_COREG_2' 
```

``` 
pazienti
├── name1
│   ├── FLAIR_name1.nii
│   ├── T1_name1.nii
│   └── T1xFLAIR.nii
├── name2
│   ├── FLAIR_name2.nii
│   ├── T1_name2.nii
│   └── T1xFLAIR.nii
``` 

After Anonymization the folder structure should look something like this: 


``` 
pazienti
├── 001
│   ├── 001_ChP_mask_FLAIR_manual_seg.nii
│   ├── 001_ChP_mask_T1_manual_seg.nii
│   ├── 001_ChP_mask_T1xFLAIR_manual_seg.nii
│   ├── 001_FLAIR.nii
│   ├── 001_FLAIR_gamma_corrected.nii
│   ├── 001_T1.nii
│   ├── 001_T1_gamma_corrected.nii
│   └── 001_T1xFLAIR.nii
├── 002
│   ├── 002_ChP_mask_FLAIR_manual_seg.nii
│   ├── 002_ChP_mask_T1_manual_seg.nii
│   ├── 002_ChP_mask_T1xFLAIR_manual_seg.nii
│   ├── 002_FLAIR.nii
│   ├── 002_FLAIR_gamma_corrected.nii
│   ├── 002_T1.nii
│   ├── 002_T1_gamma_corrected.nii
│   └── 002_T1xFLAIR.nii
├── xxx
├── ...
└── README.md
```

### Step 2:  Generate Train Test Split to have the same data split for all models used afterwards
This split generates a .txt file splitting the data into training indices and test indices. 

```
python /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/Code_data_preprocessing/step1_0_group_balanced_train_test_split.py
```

### Step 3: Create dataset structure


To generate a folder structure dependent on the model you intend to use, run the following script: 

```
python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 12 --task_name 'ChoroidPlexus_T1' --datasettype 'UMAMBA' --amount_train_subjects 78 --train_test_index_list "001,004,006,014,027,101" --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'


python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 12 --task_name 'ChoroidPlexus_T1' --datasettype 'UMAMBA' --amount_train_subjects 78 --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'

python3 step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_UMAMBA' --train_test_index_list "093,040,032,096,053,017,044,011,054,009,072,008,067,003,092,002,076,068,029,037,018,041,100,004,036,090,043,071,061,038,103,077,022,013,101,094,066,060,079,001,033,058,021,030,056,069,063,015,097,059,057,046,012,099,089,048,024,098,075,042,078,023,087,034,028,039,050,027,025,055,052,014,049,081,085,010" --datasettype 'UMAMBA' --modality 'T1' --fileending '.nii'



python3 step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 11 --task_name 'ChoroidPlexus_T1_AP' --train_test_index_list "093,040,032,096,053,017,044,011,054,009,072,008,067,003,092,002,076,068,029,037,018,041,100,004,036,090,043,071,061,038,103,077,022,013,101,094,066,060,079,001,033,058,021,030,056,069,063,015,097,059,057,046,012,099,089,048,024,098,075,042,078,023,087,034,028,039,050,027,025,055,052,014,049,081,085,010" --datasettype 'ASCHOPLEX' --modality 'T1' --fileending '.nii'

```



## Aschoplex 
### Environment 
```
python3 -m venv monai13 --system-site-packages
```

## MONAI 
### Environment 
 ```
 source envs/monai13/bin/activate

 pip install monai
 pip install nibabel
 pip install mlflow
 pip install testresources
 pip install wrapt==1.14.1
 pip install protobuf==3.20.3
 pip install fire
 pip install einops==0.8.0

 ```

```
python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 12 --task_name 'ChoroidPlexus_T1' --datasettype 'UMAMBA' --amount_train_subjects 78 --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'

python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 12 --task_name 'ChoroidPlexus_T1' --datasettype 'UMAMBA' --amount_train_subjects 78 --train_test_index_list "001,004,006,014,027,101" --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'

```

### Step 2: 
```
python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 13 --task_name 'ChoroidPlexus_T1' --datasettype 'NNUNETV2' --amount_train_subjects 78 --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'

python3 Code_data_preprocessing/step1_dataset_creator.py --path /home/linuxlia/Lia_Masterthesis/data/pazienti --task_id 13 --task_name 'ChoroidPlexus_T1' --datasettype 'NNUNETV2' --amount_train_subjects 78 --train_test_index_list "001,004,006,014,027,101" --modality 'T1' --add_id_img '' --add_id_lab '' --fileending '.nii'

```

### Step 3: Generate Datalist json



## U-Mamba 
### Environment 
linuxlia@CAD-WORKSTATION
```
 conda activate umamba
 ```

### Step 3: Generate Datalist json

#### Training only UMAMBA 
```
python Code_data_preprocessing/step2_create_json_nnunetv2.py --mode "train" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset012_ChoroidPlexus_T1" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset012_ChoroidPlexus_T1" --train_val_ratio 1.0 --num_folds 1 --datasettype "UMAMBA" --modality "['T1']"
```

#### Training and prediction UMAMBA 
```
python Code_data_preprocessing/step2_create_json_nnunetv2.py --mode "train_predict" --dataroot "/home/linuxlia/Lia_Masterthesis/data/Dataset012_ChoroidPlexus_T1" --work_dir "/home/linuxlia/Lia_Masterthesis/data/Dataset012_ChoroidPlexus_T1" --train_val_ratio 1.0 --num_folds 1 --datasettype "UMAMBA" --modality "['T1']"
```

### Train 3D U-Mamba_Bot
```
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

### Train 3D U-Mamba_Enc
```
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

### Inference U-Mamba
#### Predict testing cases with U-Mamba_Bot model
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot --disable_tta
```

#### Predict testing cases with U-Mamba_Enc model
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerUMambaEnc --disable_tta
```

#### nnunetv2 
# Setting up Paths

nnU-Net relies on environment variables to know where raw data, preprocessed data and trained model weights are stored. 
To use the full functionality of nnU-Net, the following three environment variables must be set:

1) `nnUNet_raw`: This is where you place the raw datasets. This folder will have one subfolder for each dataset names 
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique) 
dataset name. The datasets must be in nnU-Net format, see [here](dataset_format.md).

    Example tree structure:
    ```
    nnUNet_raw/Dataset001_NAME1
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    nnUNet_raw/Dataset002_NAME2
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    ```

2) `nnUNet_preprocessed`: This is the folder where the preprocessed data will be saved. The data will also be read from 
this folder during training. It is important that this folder is located on a drive with low access latency and high 
throughput (such as a nvme SSD (PCIe gen 3 is sufficient)).

3) `nnUNet_results`: This specifies where nnU-Net will save the model weights. If pretrained models are downloaded, this 
is where it will save them.

### How to set environment variables
See [here](set_environment_variables.md).

In our case: 
# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export nnUNet_raw="/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw"
export nnUNet_preprocessed="/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_preprocessed"
export nnUNet_results="/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results"
```

 Furthermore, export the U-Mamba trainers to make them accessible by imports
 ```bash 
 export PYTHONPATH=$PYTHONPATH:/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/03_U-Mamba/umamba/nnunetv2
 ```

and add it to PATH
```bash 
export PATH="/home/linuxlia/miniconda3/envs/umamba/bin:$PATH"
```


# Define the path to the Conda environment's bin directory
CONDA_BIN_PATH=/home/linuxlia/miniconda3/envs/umamba/bin

# Add the directory containing the custom trainer class to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/03_U-Mamba/umamba/nnunetv2

# Preprocessing
nnUNetv2_plan_and_preprocess -d 332 --verify_dataset_integrity

# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot

IMPORTANT: If you plan to use `nnUNetv2_find_best_configuration` (see below) add the `--npz` flag. This makes 
nnU-Net save the softmax outputs during the final validation. They are needed for that. Exported softmax
predictions are very large and therefore can take up a lot of disk space, which is why this is not enabled by default.
If you ran initially without the `--npz` flag but now require the softmax predictions, simply rerun the validation with:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```


# Inference
#nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 332 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta

!! activate umamba AND monai13!!! otherwise it wont work


 ## SAM 
 ```
 source envs/sam/bin/activate
 ```
