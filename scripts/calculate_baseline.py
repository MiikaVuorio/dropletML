import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "processed_dataset", "labels.json")

# --- naive guess ---
GUESS_X = 30.0
GUESS_Y = 30.0
GUESS_R = 0.0

def calculate_baseline_mse(json_path, use_mean=True, guess_vector=np.array([30, 30, 0])):
    """
    Loads labels from a JSON file and calculates the Mean Squared Error (MSE)
    for a fixed, baseline guess.
    
    Args:
        json_path (str): The path to the labels JSON file.
        guess_vector (np.ndarray): A NumPy array representing the fixed guess, e.g., [x, y, r].
    """
    print(f"Loading labels from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {json_path}")
        return

    true_targets = [record['target'] for record in data]
    true_targets_np = np.array(true_targets)
    
    num_samples = len(true_targets_np)
    print(f"Found {num_samples} data points.")

    if use_mean:
        guess_vector = np.mean(true_targets_np, axis=0)

    squared_errors = (true_targets_np - guess_vector)**2
    
    mean_squared_error = np.mean(squared_errors)

    mse_per_component = np.mean(squared_errors, axis=0)
    mse_x = mse_per_component[0]
    mse_y = mse_per_component[1]
    mse_r = mse_per_component[2]
    
    print("\n" + "="*40)
    print("Baseline Model Performance")
    print("="*40)
    if use_mean:
        print(f"Mean Guess (x, y, r): {guess_vector}")
    else:
        print(f"Fixed Guess (x, y, r): {guess_vector}")
    print(f"\nOverall Mean Squared Error (MSE): {mean_squared_error:.4f}")
    print("\nComponent-wise MSE:")
    print(f"  - MSE for x-coordinate: {mse_x:.4f}")
    print(f"  - MSE for y-coordinate: {mse_y:.4f}")
    print(f"  - MSE for r-coordinate: {mse_r:.4f}")
    print("="*40)


if __name__ == "__main__":
    # baseline_guess = np.array([GUESS_X, GUESS_Y, GUESS_R])
    calculate_baseline_mse(LABELS_JSON_PATH)