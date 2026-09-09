"""Typed data structures for BlazePose landmarks and video results."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional


class BlazePoseLandmark(IntEnum):
    """Indices used by the 33-landmark BlazePose topology."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass(frozen=True)
class LandmarkPoint:
    """One normalized 3D landmark with confidence values."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    def is_visible(self, threshold: float = 0.5) -> bool:
        return self.visibility >= threshold and self.presence >= threshold

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "visibility": float(self.visibility),
            "presence": float(self.presence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "LandmarkPoint":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", 0.0)),
            visibility=float(data.get("visibility", 1.0)),
            presence=float(data.get("presence", 1.0)),
        )


@dataclass
class LandmarkFrame:
    """Landmarks detected in one sampled video frame."""

    frame_index: int
    timestamp: float
    landmarks: Dict[int, LandmarkPoint] = field(default_factory=dict)

    def get(self, landmark: BlazePoseLandmark | int) -> Optional[LandmarkPoint]:
        return self.landmarks.get(int(landmark))

    def visible_points(self, threshold: float = 0.5) -> Dict[int, LandmarkPoint]:
        return {
            index: point
            for index, point in self.landmarks.items()
            if point.is_visible(threshold)
        }

    def to_dict(self) -> Dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "landmarks": {
                str(index): point.to_dict() for index, point in self.landmarks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LandmarkFrame":
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            landmarks={
                int(index): LandmarkPoint.from_dict(point)
                for index, point in data.get("landmarks", {}).items()
            },
        )


@dataclass
class BlazePoseVideoResult:
    """Complete BlazePose output for one video."""

    video_info: Dict
    frames: List[LandmarkFrame] = field(default_factory=list)

    def detected_frame_count(self) -> int:
        return sum(bool(frame.landmarks) for frame in self.frames)

    def iter_detected_frames(self) -> Iterable[LandmarkFrame]:
        return (frame for frame in self.frames if frame.landmarks)

    def to_dict(self) -> Dict:
        return {
            "video_info": self.video_info,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BlazePoseVideoResult":
        return cls(
            video_info=dict(data.get("video_info", {})),
            frames=[LandmarkFrame.from_dict(item) for item in data.get("frames", [])],
        )
