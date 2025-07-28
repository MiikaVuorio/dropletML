# dropletML
ML networks for macroscopic water droplet analysis

Will develop this readme properly later, now here is just a long command for myself, because I'll probably need to use it later

python detect.py ^
    --weights ../../../Pictures/DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../../../Pictures/DropletML/data/raw/raw_real_image/real_wetting_photo.png ^
    --name real_photo_txt_out ^
    --conf 0.6 ^
    --imgsz 640 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt