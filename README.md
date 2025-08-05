# dropletML
ML networks for macroscopic water droplet analysis

Will develop this readme properly later, now here is just a long command for myself, because I'll probably need to use it later

python detect.py ^
    --weights ../DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../DropletML/data/raw/raw_real_frames ^
    --name real_frames_out ^
    --conf 0.6 ^
    --imgsz 640 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt


Use of create_heatmap_input

python AA_CREATE_HEATMAP_INPUT.py ^
  --video_path "../data/raw/raw_real_video/RS12_C0177.MP4" ^
  --labels_dir "../runs/yolov5_runs/detect/real_frames_out/" ^
  --output_dir "../data/inputs/heatmap_resnet_samples/" ^
  --label_prefix "frame_" ^
  --start_second 36 ^
  --stride 20 ^
  --label_value 0.4
  
white plastic (acrylonitrile butadiene styrene, ABS) 0.4, transparent plastic (polyethylene terephthalate, PET) 0.5 and a silicon wafer 0.2. 