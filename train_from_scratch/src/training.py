import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
from src.models.unet2d import UNET
from src.data.dataset import MRI2D_z_data
from utils import save_checkpoint

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



# Function to train the model for one epoch
def train_epoch(model, dataloader, loss_criterion, optimizer, device):
    """
    Trains the model for one epoch using the given dataloader and loss criterion.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        loss_criterion (torch.nn.Module): The loss criterion used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        device (torch.device): The device on which the training will be performed.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for data, mask, _ in tqdm(dataloader, desc='Training', leave=False):
        data, mask = data.to(device), mask.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = loss_criterion(outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


# Function to validate the model for one epoch
def validate_epoch(model, dataloader, loss_criterion, device):
    """
    Validate the model on the validation dataset for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        loss_criterion: The loss criterion used for validation.
        device (torch.device): The device to perform validation on.

    Returns:
        float: The average loss over the validation dataset for one epoch.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data, mask, _ in tqdm(dataloader, desc='Validation', leave=False):
            data, mask = data.to(device), mask.to(device)

            outputs = model(data)
            loss = loss_criterion(outputs, mask)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def main():

    # Define your dataset and dataloader
    dat_folder = os.path.join(os.getcwd(),'ANON_DATA_INTERPOL_2D_XY_180_240')
    print("dat_folder: ", dat_folder)
    dataset = MRI2D_z_data(dat_folder)
    print("Length of dataset: ", len(dataset))
    
    # Instantiate your model
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    print(model)

    # Define the loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create LeaveOneOut cross-validator
    # The "leave-one-out" (LOO) method involves training the model on all subjects except one and then 
    # validating the model on the subject that was left out. This process is repeated for each subject in the dataset. 
    loo = LeaveOneOut() # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html

    # Training loop over epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Using LeaveOneOut cross-validator
        for train_index, val_index in loo.split(dataset):
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset = torch.utils.data.Subset(dataset, val_index)

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Training
            train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device=DEVICE)
            print(f"Training Loss: {train_loss:.4f}")

            # Validation
            val_loss = validate_epoch(model, val_dataloader, loss_fn, device=DEVICE)
            print(f"Validation Loss: {val_loss:.4f}")

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)



    if __name__ == "__main__":
        main()
