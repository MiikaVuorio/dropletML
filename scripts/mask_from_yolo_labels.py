import cv2
import numpy as np
import os

def create_mask_from_yolo(txt_path, image_shape):
    """
    Creates a binary mask from a YOLOv5 prediction .txt file.

    Args:
        txt_path (str): Path to the YOLO .txt file.
        image_shape (tuple): The (height, width) of the original image.
    
    Returns:
        A NumPy array representing the binary mask.
    """
    img_height, img_width = image_shape
    # Create a black background (a NumPy array of zeros)
    print(f"  [Function] Creating mask for image size: height={img_height}, width={img_width}")
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    print(f"  [Function] Checking for YOLO label file at: {txt_path}")

    # Check if the label file exists
    if not os.path.exists(txt_path):
        print("  [Function] *** WARNING: Label file NOT FOUND. Returning an empty (all black) mask. ***")
        return mask # Return an empty mask if no objects were detected
    print("  [Function] Label file found.")


    # Open the .txt file and read each line
    with open(txt_path, "r") as f:
        lines = f.readlines()
        # --- DEBUG: Check if the file is empty ---
        if not lines:
            print("  [Function] *** WARNING: Label file is EMPTY. Returning an empty mask. ***")
            return mask
            
        print(f"  [Function] Found {len(lines)} detected objects in the file.")


        for i, line in enumerate(lines):
            # Split the line into its components
            parts = line.strip().split()
            # We don't need the class_id for the mask
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

            # --- Denormalize the coordinates ---
            # Convert from (0-1) range back to pixel values
            w = int(width_norm * img_width)
            h = int(height_norm * img_height)
            x = int(x_center_norm * img_width - w / 2)
            y = int(y_center_norm * img_height - h / 2)

            if i == 0:
                print(f"  [Function] Box 1 (pixel coords): top-left=(x:{x}, y:{y}), size=(w:{w}, h:{h})")

            # --- Draw the filled rectangle on the mask ---
            # cv2.rectangle takes top-left (x,y) and bottom-right (x+w, y+h)
            # A thickness of -1 fills the rectangle
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
            
    return mask


print("[Main] Starting script...")

image_h, image_w = 1280, 720
yolo_txt_path = '../runs/yolov5_runs/detect/real_photo_txt_out/labels/real_wetting_photo.txt'

print(f"[Main] Set image dimensions to: height={image_h}, width={image_w}")

# 1. Create the full-size mask
full_size_mask = create_mask_from_yolo(yolo_txt_path, (image_h, image_w))

# 2. Now perform the 640x640 central crop on this mask
center_x, center_y = image_w // 2, image_h // 2
crop_size = 512
start_x = center_x - (crop_size // 2)
start_y = center_y - (crop_size // 2)

final_mask_crop = full_size_mask[start_y : start_y + crop_size, start_x : start_x + crop_size]

if np.any(final_mask_crop):
    print("[Main] The final cropped mask contains white pixels (this is good).")
else:
    print("[Main] *** WARNING: The final cropped mask is all black. The output image will be black. ***")

# 3. Save your final 640x640 mask
output_path = '../data/inputs/processed_masks/real_wetting_photo.png'
print(f"[Main] Attempting to save final mask to: {output_path}")

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    print(f"[Main] Output directory '{output_dir}' does not exist. Creating it.")
    os.makedirs(output_dir)

try:
    cv2.imwrite(output_path, final_mask_crop)
    print(f"[Main] --- Success! Mask saved to {output_path} ---")
except Exception as e:
    print(f"[Main] *** ERROR: Failed to save the image. Reason: {e} ***")
