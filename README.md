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

## MONAI 
### Environment 
 ```
 source envs/monai13/bin/activate
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





 ## SAM 
 ```
 source envs/sam/bin/activate
 ```
