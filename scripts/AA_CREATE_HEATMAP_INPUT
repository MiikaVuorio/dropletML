import cv2
import numpy as np
import os
import argparse

def create_and_save_data(video_path, labels_dir, label_file_prefix, output_dir, start_second, num_frames, crop_size):
    """
    Processes a video to generate cropped monochrome images and corresponding masks
    for a specified number of frames.

    Args:
        video_path (str): Path to the input MP4 video.
        labels_dir (str): Path to the directory containing YOLO .txt label files.
        label_file_prefix (str): The prefix for label files (e.g., 'video_frame_').
        output_dir (str): The base directory to save processed images and masks.
        start_second (int): The second in the video to start processing from.
        num_frames (int): The number of consecutive frames to process.
        crop_size (int): The edge length of the square central crop.
    """
    # --- 1. Setup and Validation ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_second * fps)

    print(f"Video Info: {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Processing {num_frames} frames starting from second {start_second} (frame ~{start_frame}).")

    if start_frame + num_frames > total_frames:
        print(f"Warning: The requested sequence ({start_frame} to {start_frame + num_frames}) "
              f"exceeds the video's total frames ({total_frames}). Processing will stop at the end.")
        num_frames = total_frames - start_frame

    # Create output directories
    image_out_dir = os.path.join(output_dir, 'images')
    mask_out_dir = os.path.join(output_dir, 'masks')
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # --- 2. Jump to the Start Frame ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # --- 3. Main Processing Loop ---
    for i in range(num_frames):
        current_frame_num = start_frame + i
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {current_frame_num}. Stopping.")
            break

        print(f"\n--- Processing frame {current_frame_num} ---")
        img_height, img_width, _ = frame.shape

        # --- 3a. Process and save the monochrome image crop ---
        center_x = img_width // 2
        center_y = img_height // 2
        start_x = center_x - (crop_size // 2)
        start_y = center_y - (crop_size // 2)

        image_crop = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        monochrome_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        image_save_path = os.path.join(image_out_dir, f"{label_file_prefix}{current_frame_num:05d}.png")
        cv2.imwrite(image_save_path, monochrome_crop)
        print(f"  Saved image to: {image_save_path}")

        # --- 3b. Create and save the corresponding mask crop ---
        label_file_path = os.path.join(labels_dir, f"{label_file_prefix}{current_frame_num:05d}.txt")
        
        full_size_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        if os.path.exists(label_file_path):
            with open(label_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                    w = int(width_norm * img_width)
                    h = int(height_norm * img_height)
                    x = int(x_center_norm * img_width - w / 2)
                    y = int(y_center_norm * img_height - h / 2)
                    cv2.rectangle(full_size_mask, (x, y), (x + w, y + h), 255, thickness=-1)
        else:
            print(f"  Warning: No label file found at {label_file_path}. Creating an empty mask.")

        mask_crop = full_size_mask[start_y : start_y + crop_size, start_x : start_x + crop_size]
        
        mask_save_path = os.path.join(mask_out_dir, f"{label_file_prefix}{current_frame_num:05d}.png")
        cv2.imwrite(mask_save_path, mask_crop)
        print(f"  Saved mask to:  {mask_save_path}")

    # --- 4. Cleanup ---
    cap.release()
    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monochrome crops and masks from a video and YOLO labels.")
    
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing the YOLO .txt label files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save the output images and masks.")
    parser.add_argument("--label_prefix", type=str, default="frame_", help="The common prefix for your label and output files (e.g., 'my_video_').")
    parser.add_argument("--start_second", type=int, default=0, help="The second in the video to start processing from.")
    parser.add_argument("--num_frames", type=int, default=25, help="The number of consecutive frames to process.")
    parser.add_argument("--crop_size", type=int, default=512, help="The size of the square central crop (e.g., 512 for 512x512).")
    
    args = parser.parse_args()

    create_and_save_data(
        video_path=args.video_path,
        labels_dir=args.labels_dir,
        label_file_prefix=args.label_prefix,
        output_dir=args.output_dir,
        start_second=args.start_second,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
    )