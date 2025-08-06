import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "inputs", "processed_dataset_seed_1", "labels.json")
IMAGES_BASE_DIR = os.path.join(BASE_DIR, "..", "data", "processed_dataset_seed_1", "images")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "deeptrack2019_style_cnn.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "inference_results")

# --- Evaluation Settings ---
SPLIT_RATIO = 0.8  # Must be the SAME ratio used during training
BATCH_SIZE = 64
NUM_IMAGES_TO_VISUALIZE = 10 # How many random examples to save

# --- PyTorch Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================================

# --- COPY THE CLASS DEFINITIONS FROM YOUR TRAINING SCRIPT ---
# In a larger project, these would live in a separate file and be imported.
class ParticleDataset(Dataset):
    def __init__(self, labels_list):
        self.labels = labels_list
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        record = self.labels[idx]
        image_path = os.path.join(IMAGES_BASE_DIR, record['image_filename'])
        original_image = cv2.imread(image_path) # Keep original for drawing
        if original_image is None:
            return self.__getitem__((idx + 1) % len(self))
        
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        target = torch.tensor(record['target'], dtype=torch.float32)
        
        # Prepare image for the model
        model_input_image = image_rgb.astype(np.float32) / 255.0
        model_input_image = torch.from_numpy(model_input_image.transpose((2, 0, 1)))
        
        # Return the original image as well for visualization
        return model_input_image, target, original_image, record['image_filename']

class DeepTrackCNN(nn.Module):
    def __init__(self):
        super(DeepTrackCNN, self).__init__()
        self.conv_base = nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.dense_top = nn.Sequential(nn.Linear(64*7*7,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,3))
    def forward(self, x):
        x = self.conv_base(x); x = torch.flatten(x, 1); x = self.dense_top(x); return x

# --- INFERENCE AND EVALUATION FUNCTION ---
def evaluate_model():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load the data and create the TEST set ---
    print(f"Loading labels from {LABELS_JSON_PATH}")
    with open(LABELS_JSON_PATH, 'r') as f:
        all_labels = json.load(f)
    
    # Recreate the exact same train/validation split as in training
    dataset_size = len(all_labels)
    train_size = int(SPLIT_RATIO * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(42) # Use the same seed!
    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size], generator=generator)
    
    # We only need the validation (test) data for evaluation
    val_labels = [all_labels[i] for i in val_indices]
    test_dataset = ParticleDataset(val_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Created test set with {len(test_dataset)} images.")

    # --- 2. Load the trained model ---
    print(f"Loading trained model from {MODEL_PATH}")
    model = DeepTrackCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode

    all_pixel_distances = []
    visualized_count = 0

    # --- 3. Run inference on the test set ---
    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for model_inputs, targets, original_images, filenames in tqdm(test_loader):
            model_inputs = model_inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Get model predictions
            predictions = model(model_inputs)
            
            # Move data back to CPU for numpy/opencv operations
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # --- Calculate pixel distance for this batch ---
            # Calculate the error for x and y coordinates only
            errors_xy = predictions_np[:, :2] - targets_np[:, :2]
            # Calculate the Euclidean distance (sqrt(dx^2 + dy^2)) for each sample
            distances = np.sqrt(np.sum(errors_xy**2, axis=1))
            all_pixel_distances.extend(distances)
            
            # --- Visualize a few random examples ---
            if visualized_count < NUM_IMAGES_TO_VISUALIZE:
                for i in range(len(original_images)):
                    if visualized_count >= NUM_IMAGES_TO_VISUALIZE: break

                    img_to_draw = original_images[i].numpy()
                    
                    # Ground Truth (Blue Circle)
                    true_x, true_y = int(targets_np[i, 0]), int(targets_np[i, 1])
                    cv2.circle(img_to_draw, (true_x, true_y), 5, (255, 0, 0), 1) # Blue circle
                    
                    # Prediction (Red Cross)
                    pred_x, pred_y = int(predictions_np[i, 0]), int(predictions_np[i, 1])
                    cv2.drawMarker(img_to_draw, (pred_x, pred_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                    
                    # Save the visualized image
                    save_path = os.path.join(OUTPUT_DIR, f"result_{filenames[i]}")
                    cv2.imwrite(save_path, img_to_draw)
                    visualized_count += 1
    
    # --- 4. Print Final Results ---
    avg_pixel_distance = np.mean(all_pixel_distances)
    std_pixel_distance = np.std(all_pixel_distances)
    max_error = np.max(all_pixel_distances)
    
    print("\n" + "="*40)
    print("Model Evaluation Complete")
    print("="*40)
    print(f"Average Pixel Distance Error: {avg_pixel_distance:.4f} pixels")
    print(f"Standard Deviation of Error: {std_pixel_distance:.4f} pixels")
    print(f"Maximum Error (Distance):   {max_error:.2f} pixels")
    print(f"Saved {visualized_count} visualization images to: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()