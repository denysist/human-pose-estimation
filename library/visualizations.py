import pathlib

import cv2 as cv
import ffmpeg
import numpy as np
from torch import nn


class Visualizer:
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

    def __init__(self, model: nn.Module | None = None) -> None:
        self.model = model

    def draw_on_image(
        self,
        input: str | pathlib.Path | np.ndarray,
        boxes: np.ndarray | None = None,
        skeleton: list[tuple[int, int]] | None = None,
        keypoints: np.ndarray | None = None,
        box_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        draw_box: bool = False,
        draw_skeleton: bool = False,
        draw_keypoint: bool = False,
        box_color: tuple[int, int, int] = (255, 0, 0),
        skeleton_color: tuple[int, int, int]
        | list[tuple[int, int, int]] = (255, 255, 255),
        keypoint_color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        line_thickness: int = 1,
        point_radius: int = 3,
        output_file: str | pathlib.Path | None = None,
        show: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not isinstance(input, np.ndarray):
            input = cv.cvtColor(
                cv.imread(str(input), cv.IMREAD_COLOR), cv.COLOR_BGR2RGB
            )
        if skeleton is None:
            skeleton = self.skeleton
        if boxes is None and keypoints is None:
            output = self.model(input=input, box_threshold=box_threshold)
        else:
            output = self.filter_output(
                boxes=boxes, keypoints=keypoints, box_threshold=box_threshold
            )
        if output is None:
            output = {
                "boxes": np.full((1, 5), np.nan),
                "keypoints": np.full((1, len(set(sum(skeleton, ()))), 3), np.nan),
            }

            return input, output
        else:
            boxes, keypoints = output["boxes"], output["keypoints"]

        if draw_box:
            input = self.draw_box(
                image=input,
                boxes=boxes,
                threshold=box_threshold,
                color=box_color,
                thickness=line_thickness,
            )
        if draw_skeleton:
            input = self.draw_skeleton(
                image=input,
                skeleton=skeleton,
                keypoints=keypoints,
                threshold=keypoint_threshold,
                color=skeleton_color,
                thickness=line_thickness,
            )
        if draw_keypoint:
            input = self.draw_keypoint(
                image=input,
                keypoints=keypoints,
                threshold=keypoint_threshold,
                color=keypoint_color,
                radius=point_radius,
            )

        if isinstance(output_file, str):
            output_file = pathlib.Path(output_file)
        if show:
            cv.imshow(
                "image" if output_file is None else output_file.stem,
                cv.cvtColor(input, cv.COLOR_RGB2BGR),
            )
            cv.waitKey(0)
            cv.destroyAllWindows()
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(output_file), cv.cvtColor(input, cv.COLOR_RGB2BGR))

        return input, output

    def draw_on_video(
        self,
        input: str | pathlib.Path,
        boxes: np.ndarray | None = None,
        skeleton: list[tuple[int, int]] | None = None,
        keypoints: np.ndarray | None = None,
        box_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        draw_box: bool = False,
        draw_skeleton: bool = False,
        draw_keypoint: bool = False,
        box_color: tuple[int, int, int] = (255, 0, 0),
        skeleton_color: tuple[int, int, int]
        | list[tuple[int, int, int]] = (255, 255, 255),
        keypoint_color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        line_thickness: int = 1,
        point_radius: int = 3,
        fps: float | None = None,
        output_file: str | pathlib.Path | None = None,
        show: bool = False,
        method: str = "ffmpeg",
    ) -> dict[str, np.ndarray]:
        arguments = dict(
            input=input,
            boxes=boxes,
            skeleton=skeleton,
            keypoints=keypoints,
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
            fps=fps,
            output_file=output_file,
            show=show,
        )
        if method == "ffmpeg":
            return self.draw_on_video_ffmpeg(**arguments)
        elif method == "opencv":
            del arguments["fps"]
            return self.draw_on_video_opencv(**arguments)

    def draw_on_video_ffmpeg(
        self,
        input: str | pathlib.Path,
        boxes: np.ndarray | None = None,
        skeleton: list[tuple[int, int]] | None = None,
        keypoints: np.ndarray | None = None,
        box_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        draw_box: bool = False,
        draw_skeleton: bool = False,
        draw_keypoint: bool = False,
        box_color: tuple[int, int, int] = (255, 0, 0),
        skeleton_color: tuple[int, int, int]
        | list[tuple[int, int, int]] = (255, 255, 255),
        keypoint_color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        line_thickness: int = 1,
        point_radius: int = 3,
        fps: float | None = None,
        output_file: str | pathlib.Path | None = None,
        show: bool = False,
    ) -> dict[str, np.ndarray]:
        input = pathlib.Path(input)
        if output_file is not None:
            output_file = pathlib.Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        input_process = ffmpeg.input(str(input))
        if fps is not None:
            input_process = input_process.filter("fps", fps=fps)
        input_process = input_process.output(
            "pipe:", format="rawvideo", pix_fmt="rgb24"
        ).run_async(pipe_stdout=True)
        probe = ffmpeg.probe(str(input))
        video_info = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        )
        height, width = video_info["height"], video_info["width"]
        if output_file is not None:
            if fps is None:
                fps = video_info["r_frame_rate"]
            video_output = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                r=fps,
            )
            if len(probe["streams"]) == 2:
                audio_output = ffmpeg.input(str(input)).audio
                video_output = ffmpeg.concat(video_output, audio_output, n=2, v=1, a=1)
            output_process = video_output.output(
                str(output_file),
                format="mp4",
                pix_fmt="yuv420p",
                vcodec="libx264",
                acodec="aac",
            ).run_async(pipe_stdin=True, overwrite_output=True)

        index = 0
        outputs = []
        while True:
            buffer = input_process.stdout.read(height * width * 3)
            if not buffer:
                break
            frame = np.frombuffer(buffer=buffer, dtype=np.uint8).reshape(
                (height, width, 3)
            )
            frame, output = self.draw_on_image(
                input=frame,
                boxes=boxes[index] if boxes is not None else boxes,
                skeleton=skeleton,
                keypoints=keypoints[index] if keypoints is not None else keypoints,
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
            )
            outputs.append(output)
            index += 1
            if show:
                cv.imshow(input.stem, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                if cv.waitKey(1) == ord("q"):
                    break
            if output_file is not None:
                output_process.stdin.write(frame.tobytes())

        outputs = {
            key: np.stack([output[key] for output in outputs], axis=0)
            for key in outputs[0].keys()
        }
        if show:
            cv.destroyAllWindows()
        if output_file is not None:
            output_process.stdin.close()
        input_process.wait()
        if output_file is not None:
            output_process.wait()

        return outputs

    def draw_on_video_opencv(
        self,
        input: str | pathlib.Path,
        boxes: np.ndarray | None = None,
        skeleton: list[tuple[int, int]] | None = None,
        keypoints: np.ndarray | None = None,
        box_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        draw_box: bool = False,
        draw_skeleton: bool = False,
        draw_keypoint: bool = False,
        box_color: tuple[int, int, int] = (255, 0, 0),
        skeleton_color: tuple[int, int, int]
        | list[tuple[int, int, int]] = (255, 255, 255),
        keypoint_color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        line_thickness: int = 1,
        point_radius: int = 3,
        output_file: str | pathlib.Path | None = None,
        show: bool = False,
    ) -> dict[str, np.ndarray]:
        input = pathlib.Path(input)
        if output_file is not None:
            output_file = pathlib.Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        input = cv.VideoCapture(str(input))
        if output_file is not None:
            height, width, fps = (
                int(input.get(cv.CAP_PROP_FRAME_HEIGHT)),
                int(input.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(input.get(cv.CAP_PROP_FPS)),
            )
            output = cv.VideoWriter(
                str(output_file), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        outputs = []
        while input.isOpened():
            index = int(input.get(cv.CAP_PROP_POS_FRAMES))
            flag, frame = input.read()
            if not flag:
                break
            frame, prediction = self.draw_on_image(
                input=cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                boxes=boxes[index] if boxes is not None else boxes,
                skeleton=skeleton,
                keypoints=keypoints[index] if keypoints is not None else keypoints,
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
            )
            outputs.append(prediction)
            if show:
                cv.imshow(input.stem, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                if cv.waitKey(1) == ord("q"):
                    break
            if output_file is not None:
                output.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        outputs = {
            key: np.stack([output[key] for output in outputs], axis=0)
            for key in outputs[0].keys()
        }
        input.release()
        if show:
            cv.destroyAllWindows()
        if output_file is not None:
            output.release()
            temporary_file = output_file.parent / f"temporary_{output_file.name}"
            # ffmpeg -i input.mp4 -vcodec libx264 -acodec aac output.mp4
            process = (
                ffmpeg.input(str(output_file))
                .output(
                    str(temporary_file),
                    format="mp4",
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    acodec="aac",
                )
                .run_async(pipe_stdin=True)
            )
            process.stdin.close()
            process.wait()
            output_file.unlink()
            temporary_file.rename(output_file)

        return outputs

    @staticmethod
    def draw_box(
        image: np.ndarray,
        boxes: np.ndarray,
        threshold: float = 0.5,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        image = image.copy()
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)

        for box in boxes:
            x1, y1, x2, y2, p = box
            if np.isnan(p) or p < threshold:
                continue

            cv.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
                cv.LINE_AA,
            )

        return image

    @staticmethod
    def draw_skeleton(
        image: np.ndarray,
        skeleton: list[tuple[int, int]],
        keypoints: np.ndarray,
        threshold: float = 0.3,
        color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 255, 255),
        thickness: int = 1,
    ) -> np.ndarray:
        image = image.copy()
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)

        for points in keypoints:
            for index, (index1, index2) in enumerate(skeleton):
                p1, p2 = points[[index1, index2], 2]
                if (np.isnan(p1) or p1 < threshold) or (np.isnan(p2) or p2 < threshold):
                    continue

                point1, point2 = points[[index1, index2], :2].astype(int)
                cv.line(
                    image,
                    point1,
                    point2,
                    color[index] if isinstance(color, list) else color,
                    thickness,
                    cv.LINE_AA,
                )

        return image

    @staticmethod
    def draw_keypoint(
        image: np.ndarray,
        keypoints: np.ndarray,
        threshold: float = 0.3,
        color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
        radius: int = 3,
    ) -> np.ndarray:
        image = image.copy()
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)

        for points in keypoints:
            for index, (x, y, p) in enumerate(points):
                if np.isnan(p) or p < threshold:
                    continue

                cv.circle(
                    image,
                    (int(x), int(y)),
                    radius,
                    color[index] if isinstance(color, list) else color,
                    -1,
                    cv.LINE_AA,
                )

        return image

    @staticmethod
    def filter_output(
        boxes: np.ndarray,
        keypoints: np.ndarray,
        box_threshold: float = 0.5,
    ) -> dict[str, np.ndarray] | None:
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)

        p = boxes[:, -1]
        condition = np.argwhere(p >= box_threshold).ravel()
        if condition.size == 0:
            return None
        else:
            return {"boxes": boxes[condition], "keypoints": keypoints[condition]}
