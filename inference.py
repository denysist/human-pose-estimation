import argparse
import pathlib


from library import inferences, visualizations


skeleton = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]
box_color = (255, 0, 0)
skeleton_color = (255, 255, 255)
keypoint_color = [
    (255, 0, 0),
    (255, 127, 0),
    (255, 127, 0),
    (255, 255, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 255, 255),
    (0, 0, 255),
    (0, 0, 255),
    (127, 0, 255),
    (127, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
    (255, 0, 127),
    (255, 0, 127),
]


def inference(
    input: str | pathlib.Path,
    output: str | pathlib.Path | None,
    detection_config: str | pathlib.Path,
    detection_checkpoint: str | pathlib.Path,
    pose_config: str | pathlib.Path,
    pose_checkpoint: str | pathlib.Path,
    box_threshold: float = 0.5,
    keypoint_threshold: float = 0.3,
    draw_box: bool = False,
    draw_skeleton: bool = False,
    draw_keypoint: bool = False,
    line_thickness: int = 1,
    point_radius: int = 3,
    fps: float | None = None,
    show: bool = False,
    device: str = "cpu",
) -> None:
    input = pathlib.Path(input)
    file_type = input.suffix.lower()
    files_types = {"image": [".jpg", ".jpeg", ".png"], "video": [".mp4"]}
    if file_type not in files_types["image"] + files_types["video"]:
        raise ValueError(f"{input.suffix} is not supported")

    model = inferences.TopDownPoseModel(
        detection_config=detection_config,
        detection_checkpoint=detection_checkpoint,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        device=device,
    )
    visualizer = visualizations.Visualizer(model=model)

    arguments = dict(
        input=input,
        skeleton=skeleton,
        box_threshold=box_threshold,
        keypoint_threshold=keypoint_threshold,
        draw_box=draw_box,
        draw_skeleton=draw_skeleton,
        draw_keypoint=draw_keypoint,
        box_color=box_color,
        skeleton_color=skeleton_color,
        keypoint_color=keypoint_color,
        line_thickness=line_thickness,
        point_radius=point_radius,
        output_file=output,
        show=show,
    )
    if file_type in files_types["image"]:
        visualizer.draw_on_image(**arguments)
    else:
        visualizer.draw_on_video(fps=fps, **arguments)


def main():
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument("--input", type=str, help="image or video file")
    parser.add_argument("--output", type=str, default=None, help="output file")
    parser.add_argument("--detection-config", type=str, help="detection config")
    parser.add_argument("--detection-checkpoint", type=str, help="detection checkpoint")
    parser.add_argument("--pose-config", type=str, help="pose config")
    parser.add_argument("--pose-checkpoint", type=str, help="pose checkpoint")
    parser.add_argument(
        "--box-threshold", type=float, default=0.5, help="box threshold"
    )
    parser.add_argument(
        "--keypoint-threshold", type=float, default=0.3, help="keypoint threshold"
    )
    parser.add_argument("--draw-box", action="store_true", help="draw box or not")
    parser.add_argument(
        "--draw-skeleton", action="store_true", help="draw skeleton or not"
    )
    parser.add_argument(
        "--draw-keypoint", action="store_true", help="draw keypoint or not"
    )
    parser.add_argument("--line-thickness", type=int, default=1, help="line thickness")
    parser.add_argument("--point-radius", type=int, default=3, help="point radius")
    parser.add_argument("--fps", type=float, default=None, help="frame per second")
    parser.add_argument("--show", action="store_true", help="show result or not")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    arguments = parser.parse_args()

    inference(**vars(arguments))


if __name__ == "__main__":
    main()
