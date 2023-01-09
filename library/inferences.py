import pathlib

import cv2 as cv
from mmdet import apis as mmdet_apis, utils as mmdet_utils
from mmpose import apis as mmpose_apis, utils as mmpose_utils
import numpy as np


class DetectionModel:
    def __init__(
        self,
        config: str | pathlib.Path,
        checkpoint: str | pathlib.Path,
        device: str = "cpu",
    ) -> None:
        mmdet_utils.register_all_modules()
        self.model = mmdet_apis.init_detector(
            config=str(config),
            checkpoint=str(checkpoint),
            device=device,
            cfg_options=None,
        )

    def __call__(
        self, input: str | pathlib.Path | np.ndarray, box_threshold: float = 0.5
    ) -> np.ndarray:
        if isinstance(input, pathlib.Path):
            input = str(input)
        elif isinstance(input, np.ndarray):
            input = cv.cvtColor(input, cv.COLOR_RGB2BGR)

        mmdet_utils.register_all_modules()
        output = mmdet_apis.inference_detector(
            model=self.model, imgs=input, test_pipeline=None
        )
        output = output.pred_instances.cpu().numpy()  # TODO: CHECK ON GPU
        boxes = np.hstack((output.bboxes, output.scores[:, np.newaxis]))
        boxes = boxes[
            np.logical_and(output.labels == 0, output.scores >= box_threshold)
        ]

        return boxes


class TopDownPoseModel:
    def __init__(
        self,
        detection_config: str | pathlib.Path,
        detection_checkpoint: str | pathlib.Path,
        pose_config: str | pathlib.Path,
        pose_checkpoint: str | pathlib.Path,
        device: str = "cpu",
    ) -> None:
        self.detection_model = DetectionModel(
            config=detection_config, checkpoint=detection_checkpoint, device=device
        )
        mmpose_utils.register_all_modules()
        self.pose_model = mmpose_apis.init_model(
            config=str(pose_config),
            checkpoint=str(pose_checkpoint),
            device=device,
            cfg_options=None,
        )

    def __call__(
        self, input: str | pathlib.Path | np.ndarray, box_threshold: float = 0.5
    ) -> dict[str, np.ndarray] | None:
        boxes = self.detection_model(input=input, box_threshold=box_threshold)
        if boxes.size == 0:
            return None

        if isinstance(input, pathlib.Path):
            input = str(input)
        elif isinstance(input, np.ndarray):
            input = cv.cvtColor(input, cv.COLOR_RGB2BGR)

        mmpose_utils.register_all_modules()
        pose_output = mmpose_apis.inference_topdown(
            model=self.pose_model, img=input, bboxes=boxes[:, :4], bbox_format="xyxy"
        )
        keypoints = []
        for instance in pose_output:
            instance = instance.pred_instances
            keypoints.append(
                np.hstack(
                    (
                        instance.keypoints.squeeze(axis=0),
                        instance.keypoint_scores.reshape((-1, 1)),
                    )
                )
            )
        keypoints = np.stack(keypoints, axis=0)

        return {"boxes": boxes, "keypoints": keypoints}
