import cv2
import numpy as np
import os
import argparse

def create_atomic_samples(video_path, labels_dir, label_prefix, output_dir, start_second, end_second, num_frames, crop_size, stride, label_value, grid_size):
    """
    Processes a video to generate atomic .npz samples, each containing a sequence
    of images, masks, and a corresponding label grid.

    Args:
        video_path (str): Path to the input MP4 video.
        labels_dir (str): Path to the directory containing YOLO .txt label files.
        label_prefix (str): The prefix for label files (e.g., 'video_frame_').
        output_dir (str): The base directory to save processed images and masks.
        start_second (int): The second in the video to start processing from.
        num_frames (int): The number of consecutive frames to process.
        crop_size (int): The edge length of the square central crop.
        label_value (float): The single floating-point value for the entire label grid.
        grid_size (int): The edge dimension of the square label grid (e.g., 16 for 16x16).
    """
    # --- 1. Setup and Validation ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame = int(start_second * fps)
    last_frame = int(end_second * fps)
    if last_frame > total_frames:
        last_frame = total_frames
        
    os.makedirs(output_dir, exist_ok=True)

    # This grid is the same for every sample from this video.
    label_grid = np.full((grid_size, grid_size), label_value, dtype=np.float32)
    print(f"Generated a {grid_size}x{grid_size} label grid with the constant value {label_value}.")

    # --- 2. Main Generation Loop ---
    sample_count = 0
    for start_frame in range(first_frame, last_frame - num_frames + 1, stride):
        print(f"\n--- Generating Sample {sample_count} (starting at frame {start_frame}) ---")
        
        image_sequence = []
        mask_sequence = []
        
        is_sample_valid = True
        for i in range(num_frames):
            current_frame_num = start_frame + i
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = cap.read()

            if not ret:
                print(f"  Warning: Failed to read frame {current_frame_num}. Ending process.")
                is_sample_valid = False
                break
            
            img_height, img_width, _ = frame.shape
            center_x, center_y = img_width // 2, img_height // 2
            start_x, start_y = center_x - (crop_size // 2), center_y - (crop_size // 2)

            image_crop = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
            monochrome_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            
            label_file_path = os.path.join(labels_dir, f"{label_prefix}{current_frame_num:05d}.txt")
            full_size_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            if os.path.exists(label_file_path):
                with open(label_file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                        w, h = int(width_norm * img_width), int(height_norm * img_height)
                        x, y = int(x_center_norm * img_width - w / 2), int(y_center_norm * img_height - h / 2)
                        cv2.rectangle(full_size_mask, (x, y), (x + w, y + h), 255, thickness=-1)

            mask_crop = full_size_mask[start_y : start_y + crop_size, start_x : start_x + crop_size]
            
            image_sequence.append(monochrome_crop)
            mask_sequence.append(mask_crop)

        if is_sample_valid:
            images_np = np.stack(image_sequence, axis=0)
            masks_np = np.stack(mask_sequence, axis=0)
            
            output_path = os.path.join(output_dir, f"sample_{sample_count:04d}.npz")
            np.savez_compressed(output_path, images=images_np, masks=masks_np, label=label_grid)
            print(f"  > Saved sample to {output_path}")
            sample_count += 1

    cap.release()
    print(f"\nProcessing complete. Generated {sample_count} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate atomic .npz data samples from a video and YOLO labels.")
    
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing the YOLO .txt label files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final .npz sample files.")
    parser.add_argument("--start_second", type=int, default=0, help="The second in the video to start processing from.")
    parser.add_argument("--end_second", type=int, default=10, help="The second in the video to end processing at.")
    parser.add_argument("--label_prefix", type=str, default="frame_", help="The common prefix for your label files.")
    parser.add_argument("--label_value", type=float, required=True, help="The ground truth slipperiness value for this video.")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of consecutive frames per sample.")
    parser.add_argument("--crop_size", type=int, default=512, help="Size of the square central crop.")
    parser.add_argument("--stride", type=int, default=10, help="Number of frames to slide forward to create the next sample.")
    parser.add_argument("--grid_size", type=int, default=16, help="The edge dimension of the label grid.") # <--- NEW
    
    args = parser.parse_args()
    create_atomic_samples(**vars(args))