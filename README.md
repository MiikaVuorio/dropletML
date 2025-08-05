# dropletML
ML networks for macroscopic water droplet analysis

Will develop this readme properly later, now here is just a long command for myself, because I'll probably need to use it later

python detect.py ^
<<<<<<< HEAD
    --weights ../DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../DropletML/data/raw/raw_real_frames ^
    --name real_frames_out ^
    --conf 0.6 ^
=======
    --weights runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source data/raw/seed1_pos/droplets_seed1.png ^
    --name seed1_txt_out ^
    --conf 0.25 ^
>>>>>>> fdfce445650ec3e35da0f4d1ff33061f0db5ffc7
    --imgsz 640 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt