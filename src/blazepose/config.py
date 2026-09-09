"""Configuration objects for BlazePose processing."""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class BlazePoseConfig:
    """Runtime options shared by the detector and video pipeline."""

    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    visibility_threshold: float = 0.5
    frame_skip: int = 1
    flip_y: bool = True

    def validate(self) -> None:
        if self.model_complexity not in {0, 1, 2}:
            raise ValueError("model_complexity must be 0, 1, or 2")
        for name in (
            "min_detection_confidence",
            "min_tracking_confidence",
            "visibility_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

    def to_dict(self) -> Dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BlazePoseConfig":
        config = cls(**data)
        config.validate()
        return config
