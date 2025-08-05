# dropletML
ML networks for macroscopic water droplet analysis

Will develop this readme properly later, now here is just a long command for myself, because I'll probably need to use it later

python detect.py ^
    --weights runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source data/raw/seed1_pos/droplets_seed1.png ^
    --name seed1_txt_out ^
    --conf 0.25 ^
    --imgsz 640 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt