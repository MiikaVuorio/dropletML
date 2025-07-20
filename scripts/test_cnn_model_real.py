
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_IMAGES_DIR = os.path.join(BASE_DIR, "..", "data", "inputs", "real_world_test_images")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "deeptrack2019_style_cnn.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "real_data_predictions")

# --- Inference Settings ---
BATCH_SIZE = 64

# --- PyTorch Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================================

# --- A NEW, SIMPLER DATASET FOR INFERENCE ONLY ---
class InferenceDataset(Dataset):
    """A PyTorch Dataset that loads images from a folder for inference."""
    
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.image_filenames)} images in {image_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # Load the original image for drawing later
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None # DataLoader will collate and skip Nones
        
        # Prepare the image for the model (same steps as in training)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        model_input_image = image_rgb.astype(np.float32) / 255.0
        model_input_image = torch.from_numpy(model_input_image.transpose((2, 0, 1)))
        
        # Return the input for the model, the original image, and its filename
        return model_input_image, original_image, filename

# --- COPY THE MODEL DEFINITION (or import it from a shared file) ---
class DeepTrackCNN(nn.Module):
    def __init__(self):
        super(DeepTrackCNN, self).__init__()
        self.conv_base = nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.dense_top = nn.Sequential(nn.Linear(64*7*7,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,3))
    def forward(self, x):
        x = self.conv_base(x); x = torch.flatten(x, 1); x = self.dense_top(x); return x

# --- A custom collate function to handle `None` values from failed image reads ---
def collate_fn(batch):
    # Filter out samples that failed to load (returned None)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None, None # Return None if the whole batch failed
    # Unzip the batch
    return torch.utils.data.dataloader.default_collate(batch)

# --- INFERENCE FUNCTION ---
def run_inference():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Create the Dataset and DataLoader ---
    inference_dataset = InferenceDataset(image_dir=REAL_IMAGES_DIR)
    # Use the custom collate_fn to handle potential bad images
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # --- 2. Load the trained model ---
    print(f"Loading trained model from {MODEL_PATH}")
    model = DeepTrackCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        return
    model.eval() # Set the model to evaluation mode

    # --- 3. Run inference and save visualized results ---
    print("\nRunning inference on new images...")
    with torch.no_grad():
        for model_inputs, original_images, filenames in tqdm(inference_loader):
            if model_inputs is None: continue # Skip if the whole batch failed to load
            
            model_inputs = model_inputs.to(DEVICE)
            
            # Get model predictions
            predictions = model(model_inputs)
            predictions_np = predictions.cpu().numpy()
            
            # Draw predictions on each image in the batch
            for i in range(len(original_images)):
                img_to_draw = original_images[i].numpy()
                filename = filenames[i]
                
                # Prediction (Red Cross)
                pred_x, pred_y = int(predictions_np[i, 0]), int(predictions_np[i, 1])
                cv2.drawMarker(img_to_draw, (pred_x, pred_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                
                # Save the visualized image
                save_path = os.path.join(OUTPUT_DIR, f"prediction_{filename}")
                cv2.imwrite(save_path, img_to_draw)
    
    print("\n" + "="*40)
    print("Inference Complete")
    print("="*40)
    print(f"Saved images with predictions to: {OUTPUT_DIR}")
    print("="*40)


if __name__ == "__main__":
    run_inference()