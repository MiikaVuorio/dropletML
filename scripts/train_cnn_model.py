import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

# --- File path config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "processed_dataset", "labels.json")
IMAGES_BASE_DIR = os.path.join(BASE_DIR, "..", "data", "processed_dataset", "images")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "deeptrack2019_style_cnn.pth")
LOGS_BASE_DIR = os.path.join(BASE_DIR, "..", "runs")


# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 20
SPLIT_RATIO = 0.8 # Use 80% of data for training, 20% for validation

# --- PyTorch Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ParticleDataset(Dataset):
    """Custom PyTorch Dataset for loading particle images and their coordinates."""
    
    def __init__(self, labels_list):
        """
        Args:
            json_path (string): Path to the json file with annotations.
        """
        self.labels = labels_list

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: (image, target) where image is the transformed image tensor
                   and target is the [x, y, r] coordinate tensor.
        """
        record = self.labels[idx]
        
        # --- Load Image ---
        # OpenCV loads images in BGR format, so we convert to RGB
        image = cv2.imread(os.path.join(IMAGES_BASE_DIR, record['image_filename']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- Load Target ---
        target = torch.tensor(record['target'], dtype=torch.float32)

        # --- Transform Image ---
        # 1. Normalize pixel values from [0, 255] to [0.0, 1.0]
        image = image.astype(np.float32) / 255.0
        # 2. Transpose the image from HxWxC (Height, Width, Channels) to CxHxW
        #    as PyTorch models expect channels first.
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        
        return image, target

# --- CNN nn module ---
class DeepTrackCNN(nn.Module):
    """
    CNN architecture inspired by the DeepTrack paper (Fig. 1a).
    It consists of a convolutional base for feature extraction and a 
    dense top for coordinate regression.
    """
    def __init__(self):
        super(DeepTrackCNN, self).__init__()
        
        # --- Convolutional Base ---
        # (3 convolutional layers, each followed by ReLU and Max-Pooling)
        self.conv_base = nn.Sequential(
            # Block 1: Input (3, 60, 60) -> Output (16, 30, 30)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Input (16, 30, 30) -> Output (32, 15, 15)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Input (32, 15, 15) -> Output (64, 7, 7)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # --- Dense Top (Regressor) ---
        # Flatten the output of the conv base to feed into the dense layers
        # The input size will be 64 channels * 7 * 7 = 3136
        self.dense_top = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3) # Output: x, y, r
        )
        
    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.conv_base(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except the batch dimension
        x = self.dense_top(x)
        return x

# --- TRAINING LOOP ---
def train_model():
    """Main function to orchestrate the model training and validation process."""

    # --- Setup for tensorboard tracking ---
    run_timestamp = str(int(time.time()))
    log_dir = os.path.join(LOGS_BASE_DIR, "deeptrack_" + run_timestamp)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log directory: {log_dir}")
    
    # --- Load data records from JSON ---
    print(f"Loading labels from {LABELS_JSON_PATH}")
    with open(LABELS_JSON_PATH, 'r') as f:
        all_labels = json.load(f)
    print(f"Loaded {len(all_labels)} total data points.")
    
    # --- Split the data records ---
    dataset_size = len(all_labels)
    train_size = int(SPLIT_RATIO * dataset_size)
    val_size = dataset_size - train_size
    print(f"Splitting data: {train_size} for training, {val_size} for validation.")

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size], generator=generator)

    # Create datasets for each split using the indices
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    train_dataset = ParticleDataset(train_labels)
    val_dataset = ParticleDataset(val_labels)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Setup Model, Loss, and Optimizer ---
    model = DeepTrackCNN().to(DEVICE)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    # --- Training & Validation Loop ---
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train() # Set the model to training mode
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            predictions = model(images)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        # --- Print Epoch Results ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")
        
        # --- Log train and validation losses ---
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (Validation Loss: {best_val_loss:.6f})")

    print("\nTraining finished.")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_model()