# dropletML
ML networks for macroscopic water droplet analysis

Will develop this readme properly later, now here is just a long command for myself, because I'll probably need to use it later

python detect.py ^
    --weights ../DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../DropletML/data/raw/seed1_pos ^
    --name real_frames_out ^
    --imgsz 640 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt

--conf 0.6 ^

Use of create_heatmap_input

python AA_CREATE_HEATMAP_INPUT.py ^
  --video_path "../data/raw/raw_real_video/RS12_C0177.MP4" ^
  --labels_dir "../runs/yolov5_runs/detect/real_frames_out/" ^
  --output_dir "../data/inputs/heatmap_resnet_samples/" ^
  --label_prefix "frame_" ^
  --start_second 36 ^
  --end_second 46 ^
  --stride 20 ^
  --label_value 0.415

python evaluate_yolo_distance.py ^
  --labels_dir "../runs/yolov5_runs/detect/seed1_out/labels" ^
  --json_path "../data/raw/seed1_pos/droplets_seed1.json" ^
  --image_width 720 ^
  --image_height 1280
  
White plastic — acrylonitrile butadiene styrene, ABS — 23.8 deg = 0.415 rad
Transparent plastic — polyethylene terephthalate, PET — 28.5 deg = 0.497 rad
silicon wafer 0.2 — 10.9 deg = 0.190 rad 