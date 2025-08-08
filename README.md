# dropletML
ML networks for macroscopic water droplet analysis


Steps of running CAH heatmap generator

1. Convert 4k video into frames using create_frames.py, e.g.

python create_frames.py ^
    --video_path "../data/raw/raw_real_video/RS13_C0174.mp4" ^
    --output_dir "../data/raw/rs13_frames" ^
    --prefix "rs13_frame_"

00:04:54 for 5316 frames of rs11
00:05:23 for 5376 frames of rs13

2. Run yolo object detection model on the frames

python detect.py ^
    --weights ../DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../DropletML/data/raw/rs13_frames ^
    --name rs13_frames ^
    --imgsz 640 ^
    --conf 0.5 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt

~00:11:30 for rs11 frames
~00:22:40 for rs13 frames
3. Use create_heatmap_input.py to create the npz files for the resnet model

python create_heatmap_input.py ^
  --video_path "../data/raw/raw_real_video/RS13_C0174.MP4" ^
  --labels_dir "../runs/yolov5_runs/detect/rs13_labels_conf_05/" ^
  --output_dir "../data/inputs/heatmap_resnet_samples/" ^
  --output_prefix "rs13_sample_" ^
  --label_prefix "rs13_frame_" ^
  --start_second 30 ^
  --end_second 110 ^
  --stride 20 ^
  --label_value 0.19

rs12 elapsed 00:22:48
rs11 elapsed 00:17:22

python evaluate_yolo_distance.py ^
  --labels_dir "../runs/yolov5_runs/detect/seed1_out/labels" ^
  --json_path "../data/raw/seed1_pos/droplets_seed1.json" ^
  --image_width 720 ^
  --image_height 1280
  
RS11 — White plastic — acrylonitrile butadiene styrene, ABS — 23.8 deg = 0.415 rad
RS12 — Transparent plastic — polyethylene terephthalate, PET — 28.5 deg = 0.497 rad
RS13 — silicon wafer 0.2 — 10.9 deg = 0.190 rad `