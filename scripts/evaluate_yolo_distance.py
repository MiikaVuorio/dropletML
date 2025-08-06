import os
import json
import numpy as np
import argparse

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def evaluate_predictions(labels_dir, json_path, image_width, image_height):
    """
    Evaluates YOLO predictions against a JSON ground truth by calculating the
    average minimum Euclidean distance.

    Args:
        labels_dir (str): Path to the directory containing YOLO .txt files.
        json_path (str): Path to the ground truth JSON file.
        image_width (int): The width of the images YOLO was run on (e.g., 1280).
        image_height (int): The height of the images YOLO was run on (e.g., 720).
    """
    # --- Load the Ground Truth Data ---
    try:
        with open(json_path, 'r') as f:
            ground_truth_data = json.load(f)
        print(f"Successfully loaded ground truth data from '{json_path}'.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Ground truth JSON file not found at '{json_path}'.")
        return
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not parse '{json_path}'. Check if it's a valid JSON file.")
        return

    all_minimum_distances = []
    total_predictions_matched = 0

    # --- Iterate Through Each Prediction File ---
    yolo_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    if not yolo_files:
        print(f"FATAL ERROR: No .txt files found in '{labels_dir}'.")
        return
        
    print(f"\nFound {len(yolo_files)} prediction files. Starting evaluation...")

    for filename in yolo_files:
        # --- Match Prediction File to Ground Truth Key ---
        # Assumes the JSON key is the filename with a .png extension
        base_name = os.path.splitext(filename)[0]
        json_key = f"{base_name}.png"

        if json_key not in ground_truth_data:
            print(f"  - Warning: No ground truth entry found for '{json_key}'. Skipping file.")
            continue
        
        # Get ground truth centers for this specific image
        gt_centers_px = ground_truth_data[json_key].get("drops_pos", [])
        
        if not gt_centers_px:
            print(f"  - Info: No ground truth droplets listed for '{json_key}'. Skipping.")
            continue

        # --- Load Predictions and Denormalize ---
        predicted_centers_px = []
        prediction_file_path = os.path.join(labels_dir, filename)
        with open(prediction_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Format: class_id x_center_norm y_center_norm ...
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                
                # Denormalize to pixel coordinates
                pixel_x = x_center_norm * image_width
                pixel_y = image_height - (y_center_norm * image_height)
                predicted_centers_px.append((pixel_x, pixel_y))
        
        if not predicted_centers_px:
            # This is not an error, just means YOLO found nothing in this frame
            continue
            
        print(f"  - Processing '{filename}': Found {len(predicted_centers_px)} predictions and {len(gt_centers_px)} ground truths.")

        # --- Find Closest GT for Each Prediction and Calculate Distance ---
        for pred_point in predicted_centers_px:
            # Calculate the distance from this prediction to all ground truths
            distances_to_all_gts = [calculate_distance(pred_point, gt_point) for gt_point in gt_centers_px]
            
            # Find the minimum distance (the best match)
            min_dist = min(distances_to_all_gts)
            all_minimum_distances.append(min_dist)
            total_predictions_matched += 1

    # --- Calculate and Report Final Statistics ---
    if not all_minimum_distances:
        print("\nEvaluation complete, but no matching predictions were found to analyze.")
        return

    average_distance = np.mean(all_minimum_distances)
    std_dev_distance = np.std(all_minimum_distances)
    max_error = np.max(all_minimum_distances)

    print("\n--- Evaluation Report ---")
    print(f"Total Predictions Analyzed: {total_predictions_matched}")
    print(f"Average Minimum Distance:   {average_distance:.2f} pixels")
    print(f"Standard Deviation:         {std_dev_distance:.2f} pixels")
    print(f"Maximum Error (Distance):   {max_error:.2f} pixels")
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO prediction accuracy against ground truth JSON.")
    
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing the YOLO .txt label files.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of the source images (for denormalization).")
    parser.add_argument("--image_height", type=int, default=720, help="Height of the source images (for denormalization).")
    
    args = parser.parse_args()

    evaluate_predictions(args.labels_dir, args.json_path, args.image_width, args.image_height)