"""Kinematic feature extraction for BlazePose sequences."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .geometry import joint_angle, midpoint, point_distance, trajectory_speed
from .types import BlazePoseLandmark, LandmarkFrame, LandmarkPoint


ANGLE_DEFINITIONS = {
    "left_elbow": (
        BlazePoseLandmark.LEFT_SHOULDER,
        BlazePoseLandmark.LEFT_ELBOW,
        BlazePoseLandmark.LEFT_WRIST,
    ),
    "right_elbow": (
        BlazePoseLandmark.RIGHT_SHOULDER,
        BlazePoseLandmark.RIGHT_ELBOW,
        BlazePoseLandmark.RIGHT_WRIST,
    ),
    "left_knee": (
        BlazePoseLandmark.LEFT_HIP,
        BlazePoseLandmark.LEFT_KNEE,
        BlazePoseLandmark.LEFT_ANKLE,
    ),
    "right_knee": (
        BlazePoseLandmark.RIGHT_HIP,
        BlazePoseLandmark.RIGHT_KNEE,
        BlazePoseLandmark.RIGHT_ANKLE,
    ),
    "left_hip": (
        BlazePoseLandmark.LEFT_SHOULDER,
        BlazePoseLandmark.LEFT_HIP,
        BlazePoseLandmark.LEFT_KNEE,
    ),
    "right_hip": (
        BlazePoseLandmark.RIGHT_SHOULDER,
        BlazePoseLandmark.RIGHT_HIP,
        BlazePoseLandmark.RIGHT_KNEE,
    ),
}


@dataclass
class BlazePoseFeatureExtractor:
    """Extract frame-level and sequence-level motion descriptors."""

    visibility_threshold: float = 0.5

    def _visible_triplet(self, frame: LandmarkFrame, indices: Tuple[int, int, int]):
        points = [frame.get(index) for index in indices]
        if any(point is None for point in points):
            return None
        if any(not point.is_visible(self.visibility_threshold) for point in points):
            return None
        return points

    def extract_joint_angles(self, frame: LandmarkFrame) -> Dict[str, Optional[float]]:
        """Calculate common elbow, knee, and hip angles."""
        angles = {}
        for name, indices in ANGLE_DEFINITIONS.items():
            points = self._visible_triplet(frame, indices)
            angles[name] = joint_angle(*points) if points else None
        return angles

    def shoulder_tilt(self, frame: LandmarkFrame) -> Optional[float]:
        left = frame.get(BlazePoseLandmark.LEFT_SHOULDER)
        right = frame.get(BlazePoseLandmark.RIGHT_SHOULDER)
        if not left or not right:
            return None
        return float(np.degrees(np.arctan2(right.y - left.y, right.x - left.x)))

    def hip_tilt(self, frame: LandmarkFrame) -> Optional[float]:
        left = frame.get(BlazePoseLandmark.LEFT_HIP)
        right = frame.get(BlazePoseLandmark.RIGHT_HIP)
        if not left or not right:
            return None
        return float(np.degrees(np.arctan2(right.y - left.y, right.x - left.x)))

    def body_center(self, frame: LandmarkFrame) -> Optional[LandmarkPoint]:
        shoulders = [
            frame.get(BlazePoseLandmark.LEFT_SHOULDER),
            frame.get(BlazePoseLandmark.RIGHT_SHOULDER),
        ]
        hips = [
            frame.get(BlazePoseLandmark.LEFT_HIP),
            frame.get(BlazePoseLandmark.RIGHT_HIP),
        ]
        if not all(shoulders + hips):
            return None
        return midpoint(midpoint(*shoulders), midpoint(*hips))

    def bilateral_symmetry(self, frame: LandmarkFrame) -> Dict[str, Optional[float]]:
        """Compare left and right limb lengths."""
        pairs = {
            "upper_arm": (
                (BlazePoseLandmark.LEFT_SHOULDER, BlazePoseLandmark.LEFT_ELBOW),
                (BlazePoseLandmark.RIGHT_SHOULDER, BlazePoseLandmark.RIGHT_ELBOW),
            ),
            "forearm": (
                (BlazePoseLandmark.LEFT_ELBOW, BlazePoseLandmark.LEFT_WRIST),
                (BlazePoseLandmark.RIGHT_ELBOW, BlazePoseLandmark.RIGHT_WRIST),
            ),
            "thigh": (
                (BlazePoseLandmark.LEFT_HIP, BlazePoseLandmark.LEFT_KNEE),
                (BlazePoseLandmark.RIGHT_HIP, BlazePoseLandmark.RIGHT_KNEE),
            ),
            "lower_leg": (
                (BlazePoseLandmark.LEFT_KNEE, BlazePoseLandmark.LEFT_ANKLE),
                (BlazePoseLandmark.RIGHT_KNEE, BlazePoseLandmark.RIGHT_ANKLE),
            ),
        }
        result = {}
        for name, (left_pair, right_pair) in pairs.items():
            left_points = [frame.get(index) for index in left_pair]
            right_points = [frame.get(index) for index in right_pair]
            if not all(left_points + right_points):
                result[name] = None
                continue
            left_length = point_distance(*left_points)
            right_length = point_distance(*right_points)
            denominator = max(left_length, right_length, 1e-8)
            result[name] = 1.0 - abs(left_length - right_length) / denominator
        return result

    def sequence_motion(self, frames: Iterable[LandmarkFrame], landmark_index: int) -> Dict[str, float]:
        """Summarize speed for one landmark across detected frames."""
        points = []
        timestamps = []
        for frame in frames:
            point = frame.get(landmark_index)
            if point and point.is_visible(self.visibility_threshold):
                points.append(point)
                timestamps.append(frame.timestamp)
        if len(points) < 2:
            return {"mean_speed": 0.0, "max_speed": 0.0, "motion_energy": 0.0}
        speed = trajectory_speed(points, timestamps)
        return {
            "mean_speed": float(np.mean(speed)),
            "max_speed": float(np.max(speed)),
            "motion_energy": float(np.mean(np.square(speed))),
        }

    def extract_frame_features(self, frame: LandmarkFrame) -> Dict:
        return {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "angles": self.extract_joint_angles(frame),
            "shoulder_tilt": self.shoulder_tilt(frame),
            "hip_tilt": self.hip_tilt(frame),
            "symmetry": self.bilateral_symmetry(frame),
        }
