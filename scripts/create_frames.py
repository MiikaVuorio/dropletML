import cv2
import os
import argparse
import time

def extract_frames(video_path, output_dir, prefix='frame_', target_height=1280, start_second=0, end_second=None):
    """
    Reads a video file and saves frames from a specified time range as individual, resized image files.

    Args:
        video_path (str): The full path to the input video file.
        output_dir (str): The path to the directory where frames will be saved.
        prefix (str): The prefix to use for saved frame filenames.
        target_height (int): The target height for the output frames (width is scaled automatically).
        start_second (int): The start time in seconds from which to begin extraction.
        end_second (int, optional): The end time in seconds at which to stop extraction. 
                                    If None, extracts to the end of the video.
    """
    timer_start = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Frames will be saved to: {output_dir}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not read video FPS. Cannot proceed with time-based extraction.")
        cap.release()
        return
        
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_height == 0:
        print("Error: Could not read video dimensions.")
        cap.release()
        return

    # Calculate start and end frames
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second is not None else total_frames - 1

    # Ensure start_frame is not out of bounds
    if start_frame >= total_frames:
        print(f"Error: Start time ({start_second}s) is after the video ends.")
        cap.release()
        return
    
    # Set video capture to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Video FPS: {fps:.2f}")
    print(f"Extracting frames from {start_frame} to {end_frame} (seconds {start_second} to {end_second or 'end'}).")

    # --- Frame Extraction ---
    aspect_ratio = original_width / original_height
    target_width = int(target_height * aspect_ratio)

    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Resizing frames to:   {target_width}x{target_height}")

    frame_count = 0
    while True:
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_pos > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Using the actual frame position for the filename ensures consistency
        filename = f"{prefix}{current_frame_pos:05d}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, resized_frame)

        frame_count += 1

        # Provide progress feedback every 100 frames so you know it's working
        if frame_count % 100 == 0:
            print(f"  ... Processed {frame_count} frames (current video frame: {current_frame_pos}) ...")

    # --- Cleanup ---
    cap.release()
    print(f"\nExtraction complete. A total of {frame_count} frames were saved.")

    timer_end = time.time()
    elapsed = timer_end - timer_start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\nExecution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and resize frames from a specific time range in a video.")
    
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted frames.")
    parser.add_argument("--prefix", type=str, default="frame_", help="The common prefix for your output frame files (e.g., 'my_video_').")
    parser.add_argument("--start_second", type=int, default=0, help="The start time in seconds from which to extract frames.")
    parser.add_argument("--end_second", type=int, default=None, help="The end time in seconds at which to stop extracting frames. If not specified, extracts until the end.")
    
    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
        start_second=args.start_second,
        end_second=args.end_second
    )