import torch 
#Provides fast and flexible image augmentation for ML 
import albumentations as A
#Converts images to PyTorch tensors.
from albumentations.pytorch import ToTensorV2
#Progress bars
from tqdm import tqdm 
import torch.nn as nn 
#Cpntains optimizer algorithms 
import torch.optim as optim 
from model import UNET

from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    check_accuracy, 
    save_predictions_as_imgs,  
)

# Hyperparameters
LEARNING_RATE = 1e-4
#Parallel computing platform and programming model that allows developers to use GPUs for general-purpose processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
#Number 
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240  
PIN_MEMORY = True
#Flag to indicate if pre-trained model should be loaded
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    #Creates a progress bar for the data loader
    loop = tqdm(loader)
    

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        #Converts targets to float, adds a channel dimension, and moves them to the specified device.
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward 
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        #backward 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        #Updates model parameters
        scaler.step(optimizer)
        scaler.update() 

        #Updates the progress bar with the current loss.
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #Depending on if there are more out_channels, CrossEntopyLoss would be employed
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR, 
        VAL_IMG_DIR,
        VAL_MASK_DIR, 
        BATCH_SIZE, 
        train_transform, 
        val_transforms,
        NUM_WORKERS, 
        PIN_MEMORY
    )

    # if LOAD_MODEL: 
        # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.amp.GradScaler()

    for epoch in range(NUM_EPOCHS): 
        #One iteration of training is defined by train_fn()
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
    
        #save model 
        checkpoint = {
            "state_dic": model.state_dict, 
            "optimizer": optimizer.state_dict
        }
        save_checkpoint(checkpoint)

        #check_accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        #Print a few examples and save to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_image/", device=DEVICE
        )

if __name__ == "__main__":
    main()

