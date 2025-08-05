import cv2
import os
import argparse

def process_image(image_path, output_path, crop_size=512):
    """
    Reads an image, takes a central crop, converts it to monochrome, 
    and saves it to the specified output path.

    Args:
        image_path (str): The full path to the input image.
        output_path (str): The full path where the processed image will be saved.
        crop_size (int): The edge length of the square central crop (e.g., 640 for 640x640).
    """
    # Read the image from the specified path
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return

    # Get image dimensions
    height, width, _ = image.shape

    # --- Step 1: Validate Dimensions ---
    # Check if the image is large enough for the crop
    if height < crop_size or width < crop_size:
        print(f"Warning: Image {os.path.basename(image_path)} is too small ({width}x{height}). Skipping.")
        return

    # --- Step 2: Perform Central Crop ---
    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the top-left corner of the crop area
    start_x = center_x - (crop_size // 2)
    start_y = center_y - (crop_size // 2)

    # Extract the 640x640 central crop
    cropped_image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    # --- Step 3: Convert to Monochrome ---
    # Use cvtColor to convert the color space from BGR to GRAY
    monochrome_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # --- Step 4: Save the Result ---
    # Save the processed image to the output path
    try:
        cv2.imwrite(output_path, monochrome_image)
        print(f"Successfully processed {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")


def main():
    """
    Main function to parse command-line arguments and process images.
    """
    parser = argparse.ArgumentParser(description="Prepare images for ML model by cropping and converting to monochrome.")
    parser.add_argument("input_path", type=str, help="Path to the input image file or directory.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where processed images will be saved.")
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found. Creating it.")
        os.makedirs(output_dir)

    # Check if the input path is a directory or a single file
    if os.path.isdir(input_path):
        print(f"Processing all supported images in directory: {input_path}")
        # Supported image formats by OpenCV
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        for filename in os.listdir(input_path):
            if filename.lower().endswith(supported_formats):
                image_file_path = os.path.join(input_path, filename)
                output_file_path = os.path.join(output_dir, filename)
                process_image(image_file_path, output_file_path)

    elif os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        filename = os.path.basename(input_path)
        output_file_path = os.path.join(output_dir, filename)
        process_image(input_path, output_file_path)

    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()