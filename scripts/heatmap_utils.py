import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def create_colorbar(height, width, vmin, vmax, colormap=cv2.COLORMAP_JET):
    """
    Creates an image of a vertical colorbar with min and max value labels.

    Args:
        height (int): The height of the colorbar image.
        width (int): The width of the colorbar image.
        vmin (float): The minimum value of the scale.
        vmax (float): The maximum value of the scale.
        colormap (int): The OpenCV colormap to use.

    Returns:
        np.ndarray: The colorbar image.
    """
    # Create a gradient array from 255 down to 0.
    # The values 0-255 will be mapped to the colormap.
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    
    # Create a blank image and apply the colormap to the gradient
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    colored_gradient = cv2.applyColorMap(gradient, colormap)
    
    # Place the colored gradient into our blank image
    colorbar[:, :] = colored_gradient

    # Add text for min and max values
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colorbar, f'{vmax:.2f}', (5, 20), font, 0.6, (255, 255, 255), 2)
    cv2.putText(colorbar, f'{vmin:.2f}', (5, height - 10), font, 0.6, (255, 255, 255), 2)
    
    return colorbar

def create_and_overlay_heatmap(grid_data, original_image, vmin=None, vmax=None, alpha=0.6, colormap=cv2.COLORMAP_JET, interpolation=cv2.INTER_LINEAR):
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
    if vmin is None:
        vmin = np.min(grid_data)
    if vmax is None:
        vmax = np.max(grid_data)
    clipped_grid = np.clip(grid_data, vmin, vmax)

    normalized_grid = 255 * (clipped_grid - vmin) / (vmax - vmin)
    normalized_grid = normalized_grid.astype(np.uint8)

    target_height, target_width = original_image.shape[:2]
    heatmap = cv2.resize(normalized_grid, (target_width, target_height), interpolation=interpolation)

    colored_heatmap = cv2.applyColorMap(heatmap, colormap)

    if len(original_image.shape) == 2:
        overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_image = original_image.copy()

    beta = 1.0 - alpha
    blended_image = cv2.addWeighted(colored_heatmap, alpha, overlay_image, 1, 0)

    return blended_image


if __name__ == '__main__':
    print("Creating hetmap for the three materials...")

    OPENCV_TO_MPL_CMAP = {
        cv2.COLORMAP_JET: 'jet',
        cv2.COLORMAP_HOT: 'hot',
        cv2.COLORMAP_COOL: 'cool',
        cv2.COLORMAP_WINTER: 'winter',
        cv2.COLORMAP_SPRING: 'spring'
    }

    COLOR_MIN = 0
    COLOR_MAX = 0.6
    SELECTED_COLORMAP = cv2.COLORMAP_JET

    file_pairs = [
        {'image': 'RS11_TEST_IMAGE.png', 'grid': 'RS11_INFERENCE.npy'},
        {'image': 'RS12_TEST_IMAGE.png', 'grid': 'RS12_INFERENCE.npy'},
        {'image': 'RS13_TEST_IMAGE.png', 'grid': 'RS13_INFERENCE.npy'}
    ]

    # --- Matplotlib Plotting Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Contact angle hysteresis (CAH) Inference Validation', fontsize=20)

    # --- Loop Through and Process Each Sample ---
    for i, (ax, files) in enumerate(zip(axes, file_pairs)):
        image = cv2.imread(files['image'], cv2.IMREAD_GRAYSCALE)
        grid = np.load(files['grid'])
        average_value = np.mean(grid)

        blended_image = create_and_overlay_heatmap(
            grid_data=grid, original_image=image, vmin=COLOR_MIN,
            vmax=COLOR_MAX, colormap=SELECTED_COLORMAP
        )

        ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
        if (i == 0):
            ax.set_title(f"ABS sample, CAH: \nAverage Inference Value: {average_value:.4f} rad")
        elif (i == 1):
            ax.set_title(f"PET sample\nAverage Value: {average_value:.4f} rad")
        elif (i == 2):
            ax.set_title(f"Silicon sample\nAverage Value: {average_value:.4f} rad")
        else:
            ax.set_title("this isn't supposed to happen")
        ax.axis('off')

    # --- Create and Add the Shared Colorbar ---
    # Adjust subplot parameters to make space for the colorbar
    fig.subplots_adjust(right=0.85)

    # Create a new axis for the colorbar on the right side of the figure
    # The list represents [left, bottom, width, height] in figure coordinates.
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])

    # Create the ScalarMappable object that the colorbar needs
    norm = plt.Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
    sm = plt.cm.ScalarMappable(cmap=OPENCV_TO_MPL_CMAP[SELECTED_COLORMAP], norm=norm)
    sm.set_array([]) # Dummy array is needed

    # Create the colorbar in the new axis
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    # Add a label to the colorbar
    cbar.set_label('CAH (radians)', fontsize=14, labelpad=15)

    # --- Save and Show ---
    output_filename = "comparison_plot_final.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined plot to '{output_filename}'")

    plt.show()