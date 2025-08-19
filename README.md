# dropletML
ML networks for macroscopic water droplet analysis


Steps of running CAH heatmap generator

1. Convert 4k video into frames using create_frames.py, e.g.

python create_frames.py ^
    --video_path "../data/raw/raw_real_video/RS13_C0174.mp4" ^
    --output_dir "../data/raw/rs13_frames" ^
    --prefix "rs13_frame_"


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


4. Run the heatmap model

python CAH_heatmap_NN.py --data_dir "../data/inputs/heatmap_resnet_samples"


--- Final Loss History ---
Training Loss per Epoch:
[0.02563400499833127, 0.00788576900862002, 0.0031651268950857532, 0.002243166506620279, 0.0018374250241322443, 0.0016419830814508412, 0.0009861094384784033, 0.0010004125724663027, 0.0015650459320265024, 0.0008007897222948184, 0.0008807224673849608, 0.002453046667263455, 0.0018693372194926875, 0.0007985555314614127, 0.0010380378150633381, 0.0012948642935953103, 0.0009295639470413637, 0.0012215695955092088, 0.0013770506954945934, 0.0030419626920775043]

Validation Loss per Epoch:
[0.0022232462807248035, 0.006694570843440791, 0.001506786301615648, 0.010957560346772274, 0.002591226932903131, 0.010385485660905639, 0.0012402721195636937, 0.00313537833862938, 0.0005968306378539031, 0.0021635377604980023, 0.003827856304512049, 0.0008457123801539031, 0.0008132779854349792, 0.0010192105847333247, 0.000598215750263383, 0.0016907806391827762, 0.0022277939637812476, 0.011105922933590287, 0.16983293771627359, 0.011330290356030066]



Times elapsed
step 1
00:04:54 for 5316 frames of rs11
00:05:23 for 5376 frames of rs13
step 2
~00:11:30 for rs11 frames
~00:22:40 for rs13 frames
step 3
rs12 elapsed 00:22:48
rs11 elapsed 00:17:22
step 4
100 epochs, 297 samples
elapsed 00:30:38



Example use for evaluating YOLO algorithms performance

python evaluate_yolo_distance.py ^
  --labels_dir "../runs/yolov5_runs/detect/seed1_out/labels" ^
  --json_path "../data/raw/seed1_pos/droplets_seed1.json" ^
  --image_width 720 ^
  --image_height 1280
  
RS11 — White plastic — acrylonitrile butadiene styrene, ABS — 23.8 deg = 0.415 rad
RS12 — Transparent plastic — polyethylene terephthalate, PET — 28.5 deg = 0.497 rad
RS13 — silicon wafer 0.2 — 10.9 deg = 0.190 rad