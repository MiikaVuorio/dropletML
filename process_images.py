import cv2
import json
import os
import numpy as np
import os
from tqdm import tqdm

# FULL_RENDER_PATH = "./renders/test/full_render.png"
RENDER_PATH = os.path.abspath("data/renders/images/droplet_pos_test.png")
DROPLET_POS_JSON_PATH = os.path.abspath("data/renders/labels.json")

# --- Output Settings ---
OUTPUT_DIR = os.path.abspath("data/processed_dataset")
IMAGE_BASE_NAME = "droplet_crop"

# --- Cropping Logic ---
CROP_WIDTH = 60
CROP_HEIGHT = 60
X_OFFSET = 180
Y_OFFSET = 460
DX_RANGE = range(0, 361, 5)
DY_RANGE = range(0, 361, 5)

def process_data():
    """
    Loads a full render and droplet positions, then crops images and
    creates a PyTorch-ready labels file.
    """
    # --- 1. Load Inputs ---
    print(f"Loading full render from: {RENDER_PATH}")
    full_image = cv2.imread(RENDER_PATH)
    if full_image is None:
        print(f"Error: Could not load image at {RENDER_PATH}")
        return

    print(f"Loading droplet positions from: {DROPLET_POS_JSON_PATH}")
    with open(DROPLET_POS_JSON_PATH, 'r') as f:
        data = json.load(f)
    all_droplet_positions = np.array(data['droplet_pos_test.png']['droplet_positions'])
    
    # --- 2. Prepare Output ---
    output_images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    print(f"Saving cropped images to: {output_images_dir}")
    
    pytorch_labels = []

    # --- 3. The Main Processing Loop ---
    for dy in tqdm(DY_RANGE, desc="Processing Rows"):
        for dx in DX_RANGE:
            # --- Define the crop window in full image coordinates ---
            x_start = X_OFFSET + dx
            y_start = Y_OFFSET + dy
            x_end = x_start + CROP_WIDTH
            y_end = y_start + CROP_HEIGHT

            # --- Crop the image using numpy slicing (OpenCV uses BGR format) ---
            # Note: Numpy slicing is [y:y_end, x:x_end]
            cropped_image = full_image[y_start:y_end, x_start:x_end]
            
            # --- Find the centremost droplet for THIS crop ---
            crop_center_x = x_start + CROP_WIDTH / 2.0
            crop_center_y = y_start + CROP_HEIGHT / 2.0
            
            # Use numpy for efficient distance calculation
            distances = np.sqrt(
                np.sum((all_droplet_positions - [crop_center_x, crop_center_y])**2, axis=1)
            )
            
            if len(distances) == 0:
                continue # No droplets to process

            closest_droplet_index = np.argmin(distances)
            r = distances[closest_droplet_index]
            closest_droplet_abs = all_droplet_positions[closest_droplet_index]
            
            # --- Calculate the droplet's position RELATIVE to the crop ---
            relative_x = closest_droplet_abs[0] - x_start
            relative_y = closest_droplet_abs[1] - y_start
            
            # --- Save the cropped image and its label ---
            output_filename = f"{IMAGE_BASE_NAME}_{dx}_{dy}.png"
            output_path = os.path.join(output_images_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
            
            # Store the label data
            pytorch_labels.append({
                "image_path": os.path.abspath(output_path), # Use absolute path for robustness
                "target": [relative_x, relative_y, r]
            })

    # --- 4. Save the Final JSON for PyTorch ---
    labels_output_path = os.path.join(OUTPUT_DIR, "labels.json")
    print(f"\nSaving final labels file to: {labels_output_path}")
    with open(labels_output_path, "w") as f:
        json.dump(pytorch_labels, f, indent=2)

    print("Processing complete!")


if __name__ == "__main__":
    process_data()