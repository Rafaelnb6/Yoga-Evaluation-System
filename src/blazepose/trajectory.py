"""Trajectory construction and YOLO-compatible export for BlazePose."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .geometry import interpolate_missing, moving_average
from .types import BlazePoseLandmark, BlazePoseVideoResult, LandmarkFrame


SELECTED_LANDMARKS = [
    (BlazePoseLandmark.LEFT_SHOULDER, "left_shoulder", "Left shoulder"),
    (BlazePoseLandmark.RIGHT_SHOULDER, "right_shoulder", "Right shoulder"),
    (BlazePoseLandmark.LEFT_WRIST, "left_wrist", "Left wrist"),
    (BlazePoseLandmark.RIGHT_WRIST, "right_wrist", "Right wrist"),
    (BlazePoseLandmark.LEFT_HIP, "left_hip", "Left hip"),
    (BlazePoseLandmark.RIGHT_HIP, "right_hip", "Right hip"),
    (BlazePoseLandmark.LEFT_ANKLE, "left_ankle", "Left ankle"),
]


@dataclass
class BlazePoseTrajectoryBuilder:
    """Accumulate frames and export the seven-joint trajectory schema."""

    visibility_threshold: float = 0.5
    frames: List[LandmarkFrame] = field(default_factory=list)

    def append(self, frame: LandmarkFrame) -> None:
        self.frames.append(frame)

    def extend(self, frames: Iterable[LandmarkFrame]) -> None:
        self.frames.extend(frames)

    def clear(self) -> None:
        self.frames.clear()

    def build(self, video_info: Dict) -> Dict:
        trajectory = {}
        for output_index, (landmark_index, key, display_name) in enumerate(SELECTED_LANDMARKS):
            joint = {
                "name": key,
                "display_name": display_name,
                "x": [],
                "y": [],
                "x_raw": [],
                "y_raw": [],
                "conf": [],
                "frame_idx": [],
            }
            for frame in self.frames:
                point = frame.get(landmark_index)
                visible = point is not None and point.is_visible(self.visibility_threshold)
                joint["x"].append(point.x if visible else None)
                joint["y"].append(point.y if visible else None)
                joint["x_raw"].append(None)
                joint["y_raw"].append(None)
                joint["conf"].append(point.visibility if point else 0.0)
                joint["frame_idx"].append(frame.frame_index)
            trajectory[output_index] = joint
        return {"video_info": dict(video_info), "trajectory": trajectory}

    def build_smoothed(self, video_info: Dict, window_size: int = 5) -> Dict:
        """Build trajectories, interpolate missing samples, and smooth coordinates."""
        result = self.build(video_info)
        for joint in result["trajectory"].values():
            joint["x"] = moving_average(interpolate_missing(joint["x"]), window_size)
            joint["y"] = moving_average(interpolate_missing(joint["y"]), window_size)
        return result

    @classmethod
    def from_video_result(
        cls,
        result: BlazePoseVideoResult,
        visibility_threshold: float = 0.5,
    ) -> "BlazePoseTrajectoryBuilder":
        builder = cls(visibility_threshold=visibility_threshold)
        builder.extend(result.frames)
        return builder
