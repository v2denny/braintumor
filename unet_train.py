'''
Trains UNet model on data.
Saves the used dataset into ClearML.
( TODO ) Initiates a task in ClearML to track all the experiment results.
'''

import cv2
import numpy as np
import os
from unet_model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from clearml import Task, Dataset


# Start ClearML Task
dataset = None
try:
    dataset = Dataset.get(dataset_name='BTds_v2', dataset_project='BrainTumor')
    print("Dataset 'BTds_v2' already exists. Using existing dataset.")
except ValueError:
    print("Creating new dataset: 'BTds_v2'")
    dataset = Dataset.create(dataset_name='BTds_v2', dataset_project='BrainTumor')
    dataset.add_files('BTds')
    dataset.upload()
    dataset.finalize()
    print("Dataset uploaded and finalized.")

task = Task.init(project_name="BrainTumor", task_name="unet_training_logs")
logger = task.get_logger()

# Paths and Variables
train_path = "BTds/train"
test_path = "BTds/test"
valid_path = "BTds/valid"
model_save_path = 'unet.pth'
COLOR_MAP = [[0, 0, 0], [255, 255, 255]]
num_classes = 2  # Bg and Tumor
input_shape = (3, 256, 256)  # Adjusted for PyTorch's channels-first format
epochs = 200
batch_size = 32

# Convert 2D mask into "layers" (each layer represents a different class)
def process_mask(mask, colormap):
    output_mask = []
    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(mask, color), axis=-1)
        output_mask.append(cmap)
    output_mask = np.stack(output_mask, axis=-1)
    # Convert to PyTorch tensor and move channels to the first dimension
    output_mask = torch.tensor(output_mask, dtype=torch.float32)
    output_mask = output_mask.permute(2, 0, 1)
    return output_mask

# Loading mask
def load(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    if "masks" not in image_path:
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = process_mask(image, COLOR_MAP)
        image = torch.argmax(image, dim=0, keepdim=True)
    return image

# Train Test Split
def train_test_split():
    train_images = []
    test_images = []
    valid_images = []
    train_masks = []
    test_masks = []
    valid_masks = []

    for path in [train_path, test_path, valid_path]:
        for file_name in os.listdir(os.path.join(path, "images")):
            file_path = os.path.join(path, "images", file_name)
            if "train" in path: train_images.append(load(file_path))
            if "test" in path: test_images.append(load(file_path))
            if "valid" in path: valid_images.append(load(file_path))
        for file_name in os.listdir(os.path.join(path, "masks")):
            file_path = os.path.join(path, "masks", file_name)
            if "train" in path: train_masks.append(load(file_path))
            if "test" in path: test_masks.append(load(file_path))
            if "valid" in path: valid_masks.append(load(file_path)) 
        
    # Convert lists to tensors
    train_images = torch.stack(train_images)
    test_images = torch.stack(test_images)
    valid_images = torch.stack(valid_images)
    train_masks = torch.stack(train_masks).squeeze(1)
    test_masks = torch.stack(test_masks).squeeze(1)
    valid_masks = torch.stack(valid_masks).squeeze(1)

    return train_images, test_images, valid_images, train_masks, test_masks, valid_masks

# Enable Nvidia GPU
def enable_GPU():
    if torch.cuda.is_available():
        print("CUDA is available. GPU enabled.")
        print("Available GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f" - Device {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    return device

def train_model(train_images, train_masks, valid_images, valid_masks, num_classes, input_shape, epochs, batch_size, model_save_path, device):
    model = UNet(input_channels=input_shape[0], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    train_dataset = TensorDataset(train_images, train_masks)
    valid_dataset = TensorDataset(valid_images, valid_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()

        # Progress bar for the training loop
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.report_scalar("Loss", "train_loss", value=loss.item(), iteration=epoch)
            
            # Update progress bar
            train_bar.set_postfix(loss=running_loss / len(train_loader))

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                logger.report_scalar("Loss", "val_loss", value=val_loss / len(valid_loader), iteration=epoch)
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        eta = epoch_time * (epochs - epoch - 1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(valid_loader):.4f}, ETA: {eta//60:.0f}m {eta%60:.0f}s")

    # Save only the state dictionary
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

# Evaluate model
def evaluate_model(model_save_path, test_images, test_masks, device):
    # Load model
    model = UNet(input_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()
    test_dataset = TensorDataset(test_images, test_masks)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == masks).sum().item()
            total += masks.numel()

    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

# Main
if __name__ == '__main__':
    print("Dataset Operations Started...")
    train_images, test_images, valid_images, train_masks, test_masks, valid_masks = train_test_split()
    print("Data Created!")
    print("Enabling GPU...")
    device = enable_GPU()
    print("Training the model (might take a while)...")
    train_model(train_images, train_masks, valid_images, valid_masks, num_classes, input_shape, epochs, batch_size, model_save_path, device)
    print("Training finished without errors!")
