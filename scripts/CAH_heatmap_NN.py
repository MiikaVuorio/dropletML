import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import argparse
import time

# --- Create a Custom Model Class ---
class HeatmapResNet(nn.Module):
    def __init__(self, in_channels=50, grid_size=16):
        super().__init__()
        
        base_resnet = models.resnet18(pretrained=True)
        self.modify_first_layer(base_resnet, in_channels)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2])
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor_head = nn.Linear(
            in_features=512,
            out_features=grid_size * grid_size
        )
        self.grid_size = grid_size

    def modify_first_layer(self, model, in_channels):
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False
        )
        with torch.no_grad():
            original_weights = model.conv1.weight.clone()
            avg_weights = torch.mean(original_weights, dim=1, keepdim=True)
            new_conv1.weight = nn.Parameter(avg_weights.repeat(1, in_channels, 1, 1))
        
        model.conv1 = new_conv1

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling_layer(x)
        x = torch.flatten(x, 1)
        x = self.regressor_head(x)
        output_grid = x.view(-1, self.grid_size, self.grid_size)

        return output_grid

class WettingDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        if not self.file_paths:
            raise FileNotFoundError(f"No .npz files found in the directory: {data_dir}")
        print(f"Found {len(self.file_paths)} samples in '{data_dir}'.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        
        images = data['images']
        masks = data['masks']
        label = data['label']
        
        # Interleave images and masks: [img0, msk0, img1, msk1, ...]
        combined_input = np.empty((images.shape[0] * 2, *images.shape[1:]), dtype=np.uint8)
        combined_input[0::2] = images
        combined_input[1::2] = masks
        
        # Convert to PyTorch tensors and normalize pixel values to 0-1 range
        combined_input_tensor = torch.from_numpy(combined_input).float() / 255.0
        label_tensor = torch.from_numpy(label).float()
        
        return combined_input_tensor, label_tensor

# --- The Main Training and Validation Functions ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, writer):
    print("\n--- Starting Training ---")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        total_train_loss = 0.0
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_train_loss += loss.item()
            if (i + 1) % 10 == 0: # Print and log progress every 10 batches
                batch_loss = running_loss / 10
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {batch_loss:.4f}")
                
                # Log batch loss to TensorBoard
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_batch', batch_loss, global_step)
                
                running_loss = 0.0

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # --- Log to TensorBoard ---
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch + 1)
        writer.add_scalar('Time/epoch_duration_seconds', epoch_duration, epoch + 1)
        
        # --- Print Epoch Summary ---
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")
        print("--------------------------\n")

    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the HeatmapResNet model on .npz data.")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the .npz sample files.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation (e.g., 0.2 for 20%).")
    
    args = parser.parse_args()

    # --- Initialize TensorBoard SummaryWriter ---
    writer = SummaryWriter()
    
    # --- Setup Device, Model, Data, Loss, and Optimizer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # The default data has 25 frames * (1 image + 1 mask) = 50 channels
    model = HeatmapResNet(in_channels=50, grid_size=16).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {total_params:,} trainable parameters.")
    
    # Load and split the dataset
    full_dataset = WettingDataset(args.data_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Loss function (Mean Squared Error is ideal for comparing grids)
    criterion = nn.MSELoss()
    
    # Optimizer (Adam is a great general-purpose choice)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Run the Training ---
    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, writer)
    finally:
        # --- Close the TensorBoard writer ---
        writer.close()

    # --- Save the Trained Model ---
    model_save_path = "heatmap_resnet_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
    print("Run `tensorboard --logdir=runs` to view the training logs.")