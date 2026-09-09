"""MediaPipe BlazePose detector wrapper."""

from typing import Optional

import cv2
import numpy as np

from .config import BlazePoseConfig
from .types import LandmarkFrame, LandmarkPoint


def _load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for BlazePose. Install dependencies from requirements.txt."
        ) from exc
    return mp


class BlazePoseDetector:
    """Stateful wrapper around MediaPipe Pose."""

    def __init__(self, config: Optional[BlazePoseConfig] = None):
        self.config = config or BlazePoseConfig()
        self.config.validate()
        self._mp = _load_mediapipe()
        self._pose = self._mp.solutions.pose.Pose(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            enable_segmentation=self.config.enable_segmentation,
            smooth_segmentation=self.config.smooth_segmentation,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe graph resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "BlazePoseDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def detect_rgb(self, image_rgb: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> LandmarkFrame:
        """Detect landmarks in one RGB image."""
        if self._pose is None:
            raise RuntimeError("Detector is closed")
        result = self._pose.process(image_rgb)
        landmarks = {}
        if result.pose_landmarks:
            for index, item in enumerate(result.pose_landmarks.landmark):
                y_value = 1.0 - float(item.y) if self.config.flip_y else float(item.y)
                landmarks[index] = LandmarkPoint(
                    x=float(item.x),
                    y=y_value,
                    z=float(item.z),
                    visibility=float(getattr(item, "visibility", 1.0)),
                    presence=float(getattr(item, "presence", 1.0)),
                )
        return LandmarkFrame(frame_index=frame_index, timestamp=timestamp, landmarks=landmarks)

    def detect_bgr(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> LandmarkFrame:
        """Detect landmarks in one OpenCV BGR image."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        return self.detect_rgb(image_rgb, frame_index=frame_index, timestamp=timestamp)

    def draw_landmarks(self, image_bgr: np.ndarray, frame: LandmarkFrame) -> np.ndarray:
        """Draw a lightweight skeleton from a typed frame."""
        output = image_bgr.copy()
        height, width = output.shape[:2]
        pose_connections = self._mp.solutions.pose.POSE_CONNECTIONS
        for first_index, second_index in pose_connections:
            first = frame.landmarks.get(first_index)
            second = frame.landmarks.get(second_index)
            if not first or not second:
                continue
            if not first.is_visible(self.config.visibility_threshold):
                continue
            if not second.is_visible(self.config.visibility_threshold):
                continue
            first_xy = (int(first.x * width), int((1.0 - first.y if self.config.flip_y else first.y) * height))
            second_xy = (int(second.x * width), int((1.0 - second.y if self.config.flip_y else second.y) * height))
            cv2.line(output, first_xy, second_xy, (80, 220, 120), 2)
        for point in frame.landmarks.values():
            if not point.is_visible(self.config.visibility_threshold):
                continue
            xy = (int(point.x * width), int((1.0 - point.y if self.config.flip_y else point.y) * height))
            cv2.circle(output, xy, 3, (30, 100, 240), -1)
        return output
