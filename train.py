import argparse
import os
import shutil

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from utils import save_checkpoint

from src.models.unet2d import UNET
from src.data.dataset import MRI2D_z_data, MRI3D_data
from src.metrics import DiceLoss, IoULoss # TODO

# Hyperparameters etc.
LEARNING_RATE = 1e-4
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 180  
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('-p', '--path', help='Path for storing checkpoints and results', required=True, default="src/results")
    parser.add_argument('-d', '--data', help='Path to data root', required=True, default="ANON_DATA_INTERPOL_2D_XY_180_240")
    parser.add_argument('-r', '--resume', help='Resume training from specified checkpoint', required=False)
    parser.add_argument('--epochs', help='Number of epochs', default=3, type=int)
    parser.add_argument('--batch-size', help='Batch Size', default=2, type=int)
    parser.add_argument('--dataloader-workers', help='Num of workers for dataloader', default=3, type=int)
    parser.add_argument('--train-val-split', help='Fraction for train/val split', default=.8, type=float)
    parser.add_argument('l', '--loss', help='Which loss function to use', default='BCE', type=str)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    path_all_model_files_root = f"{args.path}/{(args.name).upper()}/"
    training_metrics_path = path_all_model_files_root + "training_metrics/"
    training_checkpoints_path = path_all_model_files_root + "training_checkpoints/"

    if args.name == "unet2d":
            model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    print(model)
    
    if args.resume:
        all_metrics_so_far = pd.read_csv(training_metrics_path + "metrics.csv")
        trained_epochs = all_metrics_so_far["Epoch"].max() + 1
        print(f"Continue training in epoch {trained_epochs}")
        model.load_state_dict(torch.load(training_checkpoints_path + "current_best.pth"))
    else:
        print("Retrain model from scratch") 
        shutil.rmtree(path_all_model_files_root, ignore_errors=True)
        os.makedirs(path_all_model_files_root)
        os.makedirs(training_metrics_path)
        os.makedirs(training_checkpoints_path)

    

    # Define the loss function and optimizer
    if args.loss == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == "Dice":
        loss_fn = DiceLoss() # TODO
    elif args.loss == "IoU":
        loss_fn = IoULoss() # TODO
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)



    


