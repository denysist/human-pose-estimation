# image
python inference.py \
    --input data/examples/inputs/images/image.jpg \
    --output data/examples/outputs/images/image.jpg \
    --detection-config mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py \
    --detection-checkpoint checkpoints/mmdet/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    --pose-config mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    --pose-checkpoint checkpoints/mmpose/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --box-threshold 0.5 \
    --keypoint-threshold 0.3 \
    --draw-box \
    --draw-skeleton \
    --draw-keypoint \
    --line-thickness 1 \
    --point-radius 3 \
    --device cpu

# video
python inference.py \
    --input data/examples/inputs/videos/video.mp4 \
    --output data/examples/outputs/videos/video.mp4 \
    --detection-config mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py \
    --detection-checkpoint checkpoints/mmdet/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    --pose-config mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    --pose-checkpoint checkpoints/mmpose/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --box-threshold 0.5 \
    --keypoint-threshold 0.3 \
    --draw-box \
    --draw-skeleton \
    --draw-keypoint \
    --line-thickness 1 \
    --point-radius 3 \
    --device cpu
