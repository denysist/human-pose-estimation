import argparse
import pathlib
import shutil

import gradio as gr
import numpy as np
from torch import nn
import yaml

from library import inferences, visualizations


class App:
    temporary_folder = pathlib.Path("storage")

    def __init__(
        self,
        model: nn.Module,
        skeleton: list[tuple[int, int]] | None = None,
        box_color: tuple[int, int, int] = (255, 0, 0),
        skeleton_color: tuple[int, int, int]
        | list[tuple[int, int, int]] = (255, 255, 255),
        keypoint_color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        examples_folder: str | pathlib.Path | None = None,
    ) -> None:
        self.visualizer = visualizations.Visualizer(model=model)
        self.skeleton = skeleton
        self.box_color = box_color
        self.skeleton_color = skeleton_color
        self.keypoint_color = keypoint_color
        self.examples_folder = (
            pathlib.Path(examples_folder)
            if examples_folder is not None
            else examples_folder
        )

        self.image_output = None
        self.video_output = None

    def detect_on_image(
        self,
        input: np.ndarray | None,
        checkbox: list[str],
        box_threshold: float,
        keypoint_threshold: float,
        line_thickness: int,
        point_radius: int,
    ) -> np.ndarray | None:
        if input is None:
            return None

        image, self.image_output = self.visualizer.draw_on_image(
            input=input,
            skeleton=self.skeleton,
            box_threshold=box_threshold,
            keypoint_threshold=keypoint_threshold,
            draw_box="bounding box" in checkbox,
            draw_skeleton="skeleton" in checkbox,
            draw_keypoint="keypoints" in checkbox,
            box_color=self.box_color,
            skeleton_color=self.skeleton_color,
            keypoint_color=self.keypoint_color,
            line_thickness=line_thickness,
            point_radius=point_radius,
        )

        return image

    def detect_on_video(
        self,
        input: str | None,
        checkbox: list[str],
        box_threshold: float,
        keypoint_threshold: float,
        line_thickness: int,
        point_radius: int,
        fps: float | None,
    ) -> str | None:
        if input is None:
            return None

        output_file = self.temporary_folder / "output.mp4"
        self.video_output = self.visualizer.draw_on_video(
            input=input,
            skeleton=self.skeleton,
            box_threshold=box_threshold,
            keypoint_threshold=keypoint_threshold,
            draw_box="bounding box" in checkbox,
            draw_skeleton="skeleton" in checkbox,
            draw_keypoint="keypoints" in checkbox,
            box_color=self.box_color,
            skeleton_color=self.skeleton_color,
            keypoint_color=self.keypoint_color,
            line_thickness=line_thickness,
            point_radius=point_radius,
            fps=None if fps == 0 else fps,
            output_file=output_file,
        )

        return str(output_file)

    def redraw_on_image(
        self,
        input: np.ndarray | None,
        checkbox: list[str],
        box_threshold: float,
        keypoint_threshold: float,
        line_thickness: int,
        point_radius: int,
    ) -> np.ndarray | None:
        if input is None:
            return None

        image, _ = self.visualizer.draw_on_image(
            input=input,
            boxes=self.image_output["boxes"],
            skeleton=self.skeleton,
            keypoints=self.image_output["keypoints"],
            box_threshold=box_threshold,
            keypoint_threshold=keypoint_threshold,
            draw_box="bounding box" in checkbox,
            draw_skeleton="skeleton" in checkbox,
            draw_keypoint="keypoints" in checkbox,
            box_color=self.box_color,
            skeleton_color=self.skeleton_color,
            keypoint_color=self.keypoint_color,
            line_thickness=line_thickness,
            point_radius=point_radius,
        )

        return image

    def redraw_on_video(
        self,
        input: str | None,
        checkbox: list[str],
        box_threshold: float,
        keypoint_threshold: float,
        line_thickness: int,
        point_radius: int,
        fps: float | None,
    ) -> str | None:
        if input is None:
            return None

        output_file = self.temporary_folder / "output.mp4"
        self.visualizer.draw_on_video(
            input=input,
            boxes=self.video_output["boxes"],
            skeleton=self.skeleton,
            keypoints=self.video_output["keypoints"],
            box_threshold=box_threshold,
            keypoint_threshold=keypoint_threshold,
            draw_box="bounding box" in checkbox,
            draw_skeleton="skeleton" in checkbox,
            draw_keypoint="keypoints" in checkbox,
            box_color=self.box_color,
            skeleton_color=self.skeleton_color,
            keypoint_color=self.keypoint_color,
            line_thickness=line_thickness,
            point_radius=point_radius,
            fps=None if fps == 0 else fps,
            output_file=output_file,
        )

        return str(output_file)

    def run(self, host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
        gr.close_all()
        demo = gr.Blocks()
        with demo:
            gr.Markdown("<h1><center>Human Pose Estimation</center></h1>")
            gr.Markdown(
                "2D Human Pose Estimation implementation for images and videos."
            )

            with gr.Tab(label="Image"):
                with gr.Row():
                    input_image = gr.Image(
                        source="upload", tool="editor", type="numpy", label="Image"
                    ).style(height=512, width=512)
                    output_image = gr.Image(type="numpy", label="Image").style(
                        height=512, width=512
                    )
                with gr.Row():
                    image_predict = gr.Button(value="Predict")
                    image_reset = gr.Button(value="Reset")
                    image_redraw = gr.Button(value="Redraw")
                if self.examples_folder is not None:
                    gr.Examples(
                        examples=[
                            str(file)
                            for file in (self.examples_folder / "images").iterdir()
                        ],
                        inputs=input_image,
                        label="Examples",
                    )
            with gr.Tab(label="Video"):
                with gr.Row():
                    input_video = gr.Video(source="upload", format="mp4", label="Video")
                    output_video = gr.Video(format="mp4", label="Video")
                with gr.Row():
                    video_predict = gr.Button(value="Predict")
                    video_reset = gr.Button(value="Reset")
                    video_redraw = gr.Button(value="Redraw")
                if self.examples_folder is not None:
                    gr.Examples(
                        examples=[
                            str(file)
                            for file in (self.examples_folder / "videos").iterdir()
                        ],
                        inputs=input_video,
                        label="Examples",
                    )
            with gr.Tab(label="Webcam Image"):
                with gr.Row():
                    input_webcam_image = gr.Image(
                        source="webcam",
                        tool="editor",
                        type="numpy",
                        label="Image",
                        mirror_webcam=True,
                    )
                    output_webcam_image = gr.Image(type="numpy", label="Image")
                with gr.Row():
                    webcam_image_predict = gr.Button(value="Predict")
                    webcam_image_reset = gr.Button(value="Reset")
                    webcam_image_redraw = gr.Button(value="Redraw")
            with gr.Tab(label="Webcam Video"):
                with gr.Row():
                    input_webcam_video = gr.Video(
                        source="webcam", format="mp4", label="Video", mirror_webcam=True
                    )
                    output_webcam_video = gr.Video(format="mp4", label="Video")
                with gr.Row():
                    webcam_video_predict = gr.Button(value="Predict")
                    webcam_video_reset = gr.Button(value="Reset")
                    webcam_video_redraw = gr.Button(value="Redraw")

            with gr.Accordion(label="Parameters", open=True):
                with gr.Row():
                    with gr.Column():
                        checkbox = gr.CheckboxGroup(
                            choices=["bounding box", "skeleton", "keypoints"],
                            value=["bounding box", "skeleton", "keypoints"],
                            type="value",
                            label="Visualization",
                        )
                        box_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.5,
                            label="Box Threshold",
                        )
                        keypoint_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.3,
                            label="Keypoint Threshold",
                        )
                    with gr.Column():
                        line_thickness = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=1,
                            label="Line Thickness",
                        )
                        point_radius = gr.Slider(
                            minimum=1, maximum=10, step=1, value=3, label="Point Radius"
                        )
                        fps = gr.Number(
                            value=None, precision=None, label="Frame Per Second"
                        )

            image_predict.click(
                fn=self.detect_on_image,
                inputs=[
                    input_image,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                ],
                outputs=output_image,
            )
            image_reset.click(
                fn=lambda: [
                    input_image.update(value=None),
                    output_image.update(value=None),
                ],
                inputs=[],
                outputs=[input_image, output_image],
            )
            image_redraw.click(
                fn=self.redraw_on_image,
                inputs=[
                    input_image,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                ],
                outputs=output_image,
            )
            video_predict.click(
                fn=self.detect_on_video,
                inputs=[
                    input_video,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                    fps,
                ],
                outputs=output_video,
            )
            video_reset.click(
                fn=lambda: [
                    input_video.update(value=None),
                    output_video.update(value=None),
                ],
                inputs=[],
                outputs=[input_video, output_video],
            )
            video_redraw.click(
                fn=self.redraw_on_video,
                inputs=[
                    input_video,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                    fps,
                ],
                outputs=output_video,
            )
            webcam_image_predict.click(
                fn=self.detect_on_image,
                inputs=[
                    input_webcam_image,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                ],
                outputs=output_webcam_image,
            )
            webcam_image_reset.click(
                fn=lambda: [
                    input_webcam_image.update(value=None),
                    output_webcam_image.update(value=None),
                ],
                inputs=[],
                outputs=[input_webcam_image, output_webcam_image],
            )
            webcam_image_redraw.click(
                fn=self.redraw_on_image,
                inputs=[
                    input_webcam_image,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                ],
                outputs=output_webcam_image,
            )
            webcam_video_predict.click(
                fn=self.detect_on_video,
                inputs=[
                    input_webcam_video,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                    fps,
                ],
                outputs=output_webcam_video,
            )
            webcam_video_reset.click(
                fn=lambda: [
                    input_webcam_video.update(value=None),
                    output_webcam_video.update(value=None),
                ],
                inputs=[],
                outputs=[input_webcam_video, output_webcam_video],
            )
            webcam_video_redraw.click(
                fn=self.redraw_on_video,
                inputs=[
                    input_webcam_video,
                    checkbox,
                    box_threshold,
                    keypoint_threshold,
                    line_thickness,
                    point_radius,
                    fps,
                ],
                outputs=output_webcam_video,
            )

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share,
            debug=False,
            show_api=False,
        )
        if self.temporary_folder.exists():
            shutil.rmtree(self.temporary_folder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Application.")
    parser.add_argument(
        "--config", type=str, default="configs/app.yaml", help="app config"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=7860, help="port")
    parser.add_argument(
        "--share", action="store_true", help="create public link or not"
    )
    parser.add_argument(
        "--no-examples", action="store_true", help="show examples or not"
    )
    arguments = parser.parse_args()

    with open(file=arguments.config, mode="r") as file:
        config = yaml.load(file, yaml.Loader)

    model = inferences.TopDownPoseModel(**config["model"])
    app = App(
        model=model,
        examples_folder=None if arguments.no_examples else config["examples_folder"],
        **config["visualization"],
    )
    app.run(host=arguments.host, port=arguments.port, share=arguments.share)


if __name__ == "__main__":
    main()
