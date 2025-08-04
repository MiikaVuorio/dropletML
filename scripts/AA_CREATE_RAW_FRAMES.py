import cv2
import os
import argparse

def extract_frames(video_path, output_dir, prefix='frame_', target_height=1280):
    """
    Reads a video file and saves all its frames as individual, resized image files.

    Args:
        video_path (str): The full path to the input video file.
        output_dir (str): The path to the directory where frames will be saved.
        prefix (str): The prefix to use for saved frame filenames.
        target_height (int): The target height for the output frames (width is scaled automatically).
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Frames will be saved to: {output_dir}")

    # Get original dimensions to calculate the new width while maintaining aspect ratio
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if original_height == 0:
        print("Error: Could not read video dimensions.")
        cap.release()
        return

    aspect_ratio = original_width / original_height
    target_width = int(target_height * aspect_ratio)

    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Resizing frames to:   {target_width}x{target_height}")

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        filename = f"{prefix}{frame_count:05d}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, resized_frame)

        frame_count += 1

        # Provide progress feedback every 100 frames so you know it's working
        if frame_count % 100 == 0:
            print(f"  ... Processed {frame_count} frames ...")

    # --- 6. Cleanup ---
    cap.release()
    print(f"\nExtraction complete. A total of {frame_count} frames were saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and resize all frames from a video for YOLO processing.")
    
    parser.add_argument("--video-path", type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the extracted 720p frames.")
    parser.add_argument("--prefix", type=str, default="frame_", help="The common prefix for your output frame files (e.g., 'my_video_').")
    
    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        prefix=args.prefix
    )