import cv2
import json
import os
import numpy as np
import os
from tqdm import tqdm

# --- Input image locations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "real_world_test_image", "real_wetting_photo.png")

# --- Output locations ---
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "real_image_crops")
IMAGE_BASE_NAME = "droplet_crop"

# --- Cropping Logic ---
CROP_WIDTH = 60
CROP_HEIGHT = 60
X_OFFSET = 180
Y_OFFSET = 460
DX_RANGE = range(0, 361, 5)
DY_RANGE = range(0, 361, 5)
DRAW_POINTS = False
CENTER_COLOR = (0, 0, 255)
CENTER_RADIUS = 2
THICKNESS = -1

def process_data():
    """
    Loads a full render and droplet positions, then crops images and
    creates a PyTorch-ready labels file.
    """
    # --- Load Inputs ---
    print(f"Loading full render from: {IMAGE_PATH}")
    full_image = cv2.imread(IMAGE_PATH)
    if full_image is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        return

    # --- Output setup ---
    output_images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    print(f"Saving cropped images to: {output_images_dir}")

    # --- Loop for crops ---
    for dy in tqdm(DY_RANGE, desc="Processing Rows"):
        for dx in DX_RANGE:
            # --- Define the crop window in full image coordinates ---
            x_start = X_OFFSET + dx
            y_start = Y_OFFSET + dy
            x_end = x_start + CROP_WIDTH
            y_end = y_start + CROP_HEIGHT

            # --- Crop the image using numpy slicing ---
            cropped_image = full_image[y_start:y_end, x_start:x_end]
            
            # --- Save the cropped image ---
            output_filename = f"{IMAGE_BASE_NAME}_{dx}_{dy}.png"
            output_path = os.path.join(output_images_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)

    print("Processing complete!")


if __name__ == "__main__":
    process_data()