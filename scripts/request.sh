# info
curl -X GET http://127.0.0.1:8000

# image
curl -X POST http://127.0.0.1:8000/image \
    -F "input=@data/examples/inputs/images/image.jpg" \
    -F "box_threshold=0.5" \
    -F "keypoint_threshold=0.3" \
    -F "line_thickness=1" \
    -F "point_radius=3" \
    -F "draw_box=true" \
    -F "draw_skeleton=true" \
    -F "draw_keypoint=true" \
    --output image.png

# video
curl -X POST http://127.0.0.1:8000/video \
    -F "input=@data/examples/inputs/videos/video.mp4" \
    -F "box_threshold=0.5" \
    -F "keypoint_threshold=0.3" \
    -F "line_thickness=1" \
    -F "point_radius=3" \
    -F "draw_box=true" \
    -F "draw_skeleton=true" \
    -F "draw_keypoint=true" \
    -F "fps=1" \
    --output video.mp4
