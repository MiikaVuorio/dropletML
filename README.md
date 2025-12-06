# DropletML—Machine Learning Methods For Droplet Analysis
ML networks for macroscopic water droplet analysis


Steps of running CAH heatmap generator

1. Convert 4K video into frames using create_frames.py, e.g.

python create_frames.py ^
    --video_path "../data/raw/raw_real_video/RS12_C0171_TEST.mp4" ^
    --output_dir "../data/raw/rs12_test_frames" ^
    --prefix "rs12_test_frame_" ^
    --start_second 35 ^
    --end_second 40


2. Run yolo object detection model on the frames

python detect.py ^
    --weights ../DropletML/runs/yolov5_runs/train/droplet_detection_run/weights/best.pt ^
    --source ../DropletML/data/raw/rs12_test_frames ^
    --name rs12_test_conf_06 ^
    --imgsz 640 ^
    --conf 0.6 ^
    --hide-labels ^
    --line-thickness 1 ^
    --save-txt


3. Use create_heatmap_input.py to create the npz files for the resnet model

python create_heatmap_input.py ^
  --video_path "../data/raw/raw_real_video/RS12_C0171_TEST.MP4" ^
  --labels_dir "../../yolov5/runs/detect/rs12_test_conf_06/" ^
  --output_dir "../data/inputs/heatmap_resnet_samples/rs12_test_samples/" ^
  --output_prefix "rs12_test_sample_" ^
  --label_prefix "rs12_test_frame_" ^
  --start_second 36 ^
  --end_second 39 ^
  --stride 20 ^
  --label_value 0.19


4. Train the heatmap model

python CAH_heatmap_NN.py ^
  --data_dir "../data/inputs/heatmap_resnet_samples"

5. Run inference with model

python CAH_model_inference.py ^
  --model_path "heatmap_resnet_final.pth" ^
  --npz_path "../data/inputs/heatmap_resnet_samples/rs13_test_samples/rs13_test_sample_0000.npz" ^
  --output_name "RS13_INFERENCE"

6. Create 512x512 crops for plots

python monochrome_image_crop.py ^
  --input_path "../data/raw/rs13_test_frames/rs13_test_frame_01000.png" ^
  --output_dir "RS13_TEST_IMAGE"


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
  
RS11 — White plastic — acrylonitrile butadiene styrene, ABS — 23.8 deg = 0.415 rad, conf=0.6
RS12 — Transparent plastic — polyethylene terephthalate, PET — 28.5 deg = 0.497 rad, conf=0.6
RS13 — silicon wafer 0.2 — 10.9 deg = 0.190 rad, conf=0.5
