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