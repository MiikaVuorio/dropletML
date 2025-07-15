import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# --- CONFIGURATION ---
# --- Input file paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_IMAGE_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "pos_and_bound", "images", "droplet_pos_test.png")
SOURCE_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "pos_and_bound", "labels.json")

# --- Output file path ---
YOLO_DATASET_DIR = os.path.join(BASE_DIR, "..", "data", "yolo_dataset")

# Data Generation Settings
NUM_CROPS = 2000  # Generate 2000 training images from the single large one
CROP_SIZE = 320   # The size of each crop (e.g., 320x320)
SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

def generate_yolo_dataset():
    # 1. Load source data
    print("Loading source data...")
    full_image = cv2.imread(SOURCE_IMAGE_PATH)
    if full_image is None:
        raise FileNotFoundError(f"Could not load image at {SOURCE_IMAGE_PATH}")
    
    img_h, img_w, _ = full_image.shape
    print(f"Loaded image of size {img_w}x{img_h}")
    
    with open(SOURCE_JSON_PATH, 'r') as f:
        all_droplets_data = json.load(f)
    
    # 2. Prepare YOLO directory structure
    train_img_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
    val_img_dir = os.path.join(YOLO_DATASET_DIR, "images", "val")
    train_label_dir = os.path.join(YOLO_DATASET_DIR, "labels", "train")
    val_label_dir = os.path.join(YOLO_DATASET_DIR, "labels", "val")
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(d, exist_ok=True)

    # 3. Generate random crop coordinates
    crop_coords = []
    for _ in range(NUM_CROPS):
        x_start = np.random.randint(0, img_w - CROP_SIZE)
        y_start = np.random.randint(0, img_h - CROP_SIZE)
        crop_coords.append((x_start, y_start))

    # 4. Split crops into training and validation sets
    train_crops, val_crops = train_test_split(crop_coords, train_size=SPLIT_RATIO, random_state=42)
    
    # 5. Process each split to generate images and labels
    for split_name, crops, img_dest, label_dest in [
        ("train", train_crops, train_img_dir, train_label_dir),
        ("val", val_crops, val_img_dir, val_label_dir)
    ]:
        print(f"Processing {split_name} set ({len(crops)} images)...")
        for i, (x_start, y_start) in enumerate(tqdm(crops)):
            x_end = x_start + CROP_SIZE
            y_end = y_start + CROP_SIZE
            
            # Create and save the cropped image
            cropped_image = full_image[y_start:y_end, x_start:x_end]
            image_filename = f"{split_name}_crop_{i}.png"
            cv2.imwrite(os.path.join(img_dest, image_filename), cropped_image)
            
            # Find droplets within this crop and create the YOLO label file
            yolo_labels = []
            for droplet_record in all_droplets_data:
                center_x_abs, center_y_abs = droplet_record['center_xy'][0], (1280 - droplet_record['center_xy'][1])
                bbox_w_abs, bbox_h_abs = droplet_record['bbox_wh'][0]*2, droplet_record['bbox_wh'][1]*2 
                
                # Check if the droplet center is inside our crop window
                if x_start < center_x_abs < x_end and y_start < center_y_abs < y_end:
                    # Convert absolute droplet coordinates to coordinates relative to the crop
                    relative_x = center_x_abs - x_start
                    relative_y = center_y_abs - y_start
                    
                    # Normalize for YOLO format (relative to the CROP_SIZE)
                    x_center_norm = relative_x / CROP_SIZE
                    y_center_norm = relative_y / CROP_SIZE
                    width_norm = bbox_w_abs / CROP_SIZE
                    height_norm = bbox_h_abs / CROP_SIZE
                    
                    class_index = 0
                    yolo_labels.append(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

            # Write the label file for this crop
            txt_filename = os.path.splitext(image_filename)[0] + ".txt"
            with open(os.path.join(label_dest, txt_filename), 'w') as f:
                f.write("\n".join(yolo_labels))
                
    # 6. Create the dataset.yaml file that YOLO needs
    yaml_data = {
        'path': os.path.abspath(YOLO_DATASET_DIR), # Tells YOLO the root of the dataset
        'train': 'images/train',  # Path relative to 'path'
        'val': 'images/val',      # Path relative to 'path'
        'nc': 1,
        'names': ['droplet']
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, "droplet_dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"\nYOLO dataset created successfully.")
    print(f"YAML configuration file at: {yaml_path}")


if __name__ == "__main__":
    generate_yolo_dataset()