# Activate the Conda environment
source /home/linuxlia/miniconda3/bin/activate umamba

# Print the Python interpreter being used
which python

which nibabel

# Print the Python version
python --version

python import nibabel

# Define the path to the Conda environment's bin directory
CONDA_BIN_PATH=/home/linuxlia/miniconda3/envs/umamba/bin


# Add the directory containing the custom trainer class to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/03_U-Mamba/umamba/nnunetv2


# Preprocessing
nnUNetv2_plan_and_preprocess -d 332 --verify_dataset_integrity



# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot



# Inference
#nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 332 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta