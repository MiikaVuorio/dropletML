import cv2
import numpy as np
from scipy import ndimage

def create_and_overlay_heatmap(grid_data, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET, interpolation=cv2.INTER_LINEAR):
    """
    Generates a heatmap from a grid of data and overlays it on an original image.

    Args:
        grid_data (np.ndarray): A 2D NumPy array (e.g., 16x16) containing the model's output.
                                  Values can be in any range.
        original_image (np.ndarray): The original image (e.g., 640x640) on which to overlay the heatmap.
                                       Can be grayscale or color.
        alpha (float): The transparency of the heatmap overlay. 0.0 is fully transparent, 1.0 is fully opaque.
        colormap (int): The OpenCV colormap to use (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_HOT).
        interpolation (int): The interpolation method for resizing the heatmap (e.g., cv2.INTER_LINEAR for smooth,
                               cv2.INTER_NEAREST for blocky).

    Returns:
        np.ndarray: The original image blended with the colored heatmap.
    """
    # --- Step 1: Normalize the grid data to the 0-255 range ---
    # This is crucial for applying a colormap.
    normalized_grid = cv2.normalize(grid_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- Step 2: Resize the small grid to match the original image's dimensions ---
    target_height, target_width = original_image.shape[:2]
    heatmap = cv2.resize(normalized_grid, (target_width, target_height), interpolation=interpolation)

    # --- Step 3: Apply the colormap to create a colored heatmap ---
    # The result is a 3-channel (BGR) image.
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)

    # --- Step 4: Prepare the original image for blending ---
    # If the original image is grayscale, convert it to BGR to match the heatmap.
    if len(original_image.shape) == 2:
        overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_image = original_image.copy()

    # --- Step 5: Blend the heatmap and the original image ---
    # The formula is: blended_img = heatmap * alpha + original_img * (1-alpha)
    beta = 1.0 - alpha
    blended_image = cv2.addWeighted(colored_heatmap, alpha, overlay_image, 1, 0)

    return blended_image


if __name__ == '__main__':
    # This block demonstrates how to use the function.
    # It will only run when you execute this script directly.
    print("Running heatmap utility demonstration...")

    # --- Create Dummy Data ---
    # 1. A dummy model output grid (16x16)
    grid_size = 16
    dummy_grid = np.random.normal(0,1,(grid_size, grid_size))
    smooth_grid = ndimage.gaussian_filter(dummy_grid, sigma=5)
    print(f"Created a dummy {grid_size}x{grid_size} grid of random values.")

    # 2. A dummy original image (e.g., 640x640)
    image_size = 512
    image_path = '../data/inputs/MONOCHROME/real_wetting_photo.png'
    real_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # --- Use the Function ---
    # Create a smooth, JET-colored heatmap
    smooth_blended_image = create_and_overlay_heatmap(smooth_grid, real_image, alpha=0.2, colormap=cv2.COLORMAP_JET, interpolation=cv2.INTER_LINEAR)
    
    # --- Display the Results ---
    # Concatenate images for easy comparison
    comparison_top = np.hstack([cv2.cvtColor(real_image, cv2.COLOR_GRAY2BGR), smooth_blended_image])
    final_display = np.vstack([comparison_top])

    cv2.imshow('Demonstration: Original (left) vs. Blended (right)', final_display)
    
    print("\nDisplaying results. Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Save an example output ---
    output_filename = "heatmap_example.png"
    cv2.imwrite(output_filename, smooth_blended_image)
    print(f"Saved an example heatmap to '{output_filename}'.")