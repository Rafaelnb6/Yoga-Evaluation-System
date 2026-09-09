"""Geometry and temporal helpers for BlazePose landmarks."""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import BlazePoseLandmark, LandmarkFrame, LandmarkPoint


def point_distance(first: LandmarkPoint, second: LandmarkPoint, use_3d: bool = False) -> float:
    """Return Euclidean distance between two landmarks."""
    dz = first.z - second.z if use_3d else 0.0
    return float(math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + dz ** 2))


def midpoint(first: LandmarkPoint, second: LandmarkPoint) -> LandmarkPoint:
    """Return the component-wise midpoint of two landmarks."""
    return LandmarkPoint(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        z=(first.z + second.z) / 2.0,
        visibility=min(first.visibility, second.visibility),
        presence=min(first.presence, second.presence),
    )


def joint_angle(first: LandmarkPoint, center: LandmarkPoint, third: LandmarkPoint) -> float:
    """Return the angle in degrees at the center landmark."""
    vector_a = np.array([first.x - center.x, first.y - center.y, first.z - center.z])
    vector_b = np.array([third.x - center.x, third.y - center.y, third.z - center.z])
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def torso_center(frame: LandmarkFrame) -> Optional[LandmarkPoint]:
    """Estimate torso center from shoulder and hip midpoints."""
    left_shoulder = frame.get(BlazePoseLandmark.LEFT_SHOULDER)
    right_shoulder = frame.get(BlazePoseLandmark.RIGHT_SHOULDER)
    left_hip = frame.get(BlazePoseLandmark.LEFT_HIP)
    right_hip = frame.get(BlazePoseLandmark.RIGHT_HIP)
    if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
        return None
    return midpoint(midpoint(left_shoulder, right_shoulder), midpoint(left_hip, right_hip))


def torso_scale(frame: LandmarkFrame) -> float:
    """Estimate a scale factor from shoulder and hip widths."""
    pairs = [
        (BlazePoseLandmark.LEFT_SHOULDER, BlazePoseLandmark.RIGHT_SHOULDER),
        (BlazePoseLandmark.LEFT_HIP, BlazePoseLandmark.RIGHT_HIP),
    ]
    distances = []
    for first_index, second_index in pairs:
        first = frame.get(first_index)
        second = frame.get(second_index)
        if first and second:
            distances.append(point_distance(first, second))
    return float(np.mean(distances)) if distances else 1.0


def normalize_frame(frame: LandmarkFrame, center_on_torso: bool = True) -> LandmarkFrame:
    """Center and scale one frame while preserving confidence values."""
    center = torso_center(frame) if center_on_torso else LandmarkPoint(0.0, 0.0, 0.0)
    center = center or LandmarkPoint(0.0, 0.0, 0.0)
    scale = max(torso_scale(frame), 1e-8)
    landmarks = {
        index: LandmarkPoint(
            x=(point.x - center.x) / scale,
            y=(point.y - center.y) / scale,
            z=(point.z - center.z) / scale,
            visibility=point.visibility,
            presence=point.presence,
        )
        for index, point in frame.landmarks.items()
    }
    return LandmarkFrame(frame.frame_index, frame.timestamp, landmarks)


def mirror_frame(frame: LandmarkFrame) -> LandmarkFrame:
    """Mirror normalized X coordinates around the image center."""
    return LandmarkFrame(
        frame_index=frame.frame_index,
        timestamp=frame.timestamp,
        landmarks={
            index: LandmarkPoint(
                x=1.0 - point.x,
                y=point.y,
                z=point.z,
                visibility=point.visibility,
                presence=point.presence,
            )
            for index, point in frame.landmarks.items()
        },
    )


def visibility_ratio(frame: LandmarkFrame, threshold: float = 0.5) -> float:
    """Return the fraction of landmarks above a visibility threshold."""
    if not frame.landmarks:
        return 0.0
    visible = sum(point.is_visible(threshold) for point in frame.landmarks.values())
    return visible / len(frame.landmarks)


def interpolate_missing(values: Sequence[Optional[float]]) -> List[float]:
    """Fill missing scalar samples with linear interpolation."""
    if not values:
        return []
    array = np.array([np.nan if value is None else float(value) for value in values])
    valid = np.flatnonzero(~np.isnan(array))
    if not len(valid):
        return [0.0] * len(values)
    missing = np.flatnonzero(np.isnan(array))
    array[missing] = np.interp(missing, valid, array[valid])
    return array.tolist()


def moving_average(values: Sequence[float], window_size: int = 5) -> List[float]:
    """Smooth a scalar sequence with a centered moving average."""
    if window_size <= 1 or len(values) < window_size:
        return [float(value) for value in values]
    radius = window_size // 2
    return [
        float(np.mean(values[max(0, index - radius): index + radius + 1]))
        for index in range(len(values))
    ]


def finite_difference(values: Sequence[float], timestamps: Sequence[float]) -> List[float]:
    """Estimate the first temporal derivative."""
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have equal lengths")
    if len(values) < 2:
        return [0.0] * len(values)
    derivative = np.gradient(np.asarray(values, dtype=float), np.asarray(timestamps, dtype=float))
    return derivative.tolist()


def trajectory_speed(points: Sequence[LandmarkPoint], timestamps: Sequence[float]) -> List[float]:
    """Estimate 2D point speed across a sequence."""
    if len(points) != len(timestamps):
        raise ValueError("points and timestamps must have equal lengths")
    if len(points) < 2:
        return [0.0] * len(points)
    x_speed = finite_difference([point.x for point in points], timestamps)
    y_speed = finite_difference([point.y for point in points], timestamps)
    return [float(math.hypot(x_value, y_value)) for x_value, y_value in zip(x_speed, y_speed)]


def bounding_box(frame: LandmarkFrame, threshold: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    """Return the normalized visible-landmark bounding box."""
    points = list(frame.visible_points(threshold).values())
    if not points:
        return None
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def select_landmarks(frame: LandmarkFrame, indices: Iterable[int]) -> Dict[int, LandmarkPoint]:
    """Return a subset of landmarks that exist in a frame."""
    return {int(index): frame.landmarks[int(index)] for index in indices if int(index) in frame.landmarks}
