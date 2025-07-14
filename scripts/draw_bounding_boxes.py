import cv2
import json
import os
import numpy as np

# --- Input file paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "pos_and_bound", "images", "droplet_pos_test.png")
JSON_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "pos_and_bound", "labels.json")

# --- Output file path ---
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "bounding_boxes")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "render_with_bboxes.png")

# --- Drawing Settings ---
BOX_COLOR = (0, 255, 0)  # Green in BGR format
BOX_THICKNESS = 1       # 1 pixel thick
# =========================================================================


def draw_bounding_boxes():
    """
    Loads a source image and a JSON file with bounding box data,
    draws the boxes on the image, and saves/displays the result.
    """
    # --- Load the source image ---
    print(f"Loading source image from: {RENDER_PATH}")
    image = cv2.imread(RENDER_PATH)
    if image is None:
        print(f"FATAL ERROR: Could not find or read the source image at '{RENDER_PATH}'")
        return

    # --- Load the JSON data ---
    print(f"Loading bounding box data from: {JSON_DATA_PATH}")
    try:
        with open(JSON_DATA_PATH, 'r') as f:
            droplet_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the JSON file at '{JSON_DATA_PATH}'")
        return
        
    print(f"Found data for {len(droplet_data)} droplets. Drawing boxes...")

    # --- Loop through each droplet record and draw the box ---
    for record in droplet_data:
        center_xy = record['center_xy']
        bbox_wh = record['bbox_wh']
        
        center_x, center_y = center_xy[0], center_xy[1]
        width, height = bbox_wh[0], bbox_wh[1]
        
        # --- Convert from (center, width, height) to (top-left, bottom-right) ---
        top_left_x = int(center_x - width / 2)
        top_left_y = int(center_y - height / 2)
        bottom_right_x = int(center_x + width / 2)
        bottom_right_y = int(center_y + height / 2)
        
        # --- Draw the rectangle on the image ---
        cv2.rectangle(
            image, 
            (top_left_x, top_left_y), 
            (bottom_right_x, bottom_right_y), 
            BOX_COLOR, 
            BOX_THICKNESS
        )

    # --- 4. Save the final image ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"\nSuccess! Image with bounding boxes saved to: {OUTPUT_IMAGE_PATH}")
    
    # --- 5. Display the image in a window ---
    print("Displaying image. Press any key to close the window.")
    cv2.namedWindow("Bounding Box Verification", cv2.WINDOW_NORMAL)
    cv2.imshow("Bounding Box Verification", image)
    
    # Wait for the user to press a key, then clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    draw_bounding_boxes()