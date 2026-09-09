"""Quality metrics for BlazePose detections and trajectories."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np

from .geometry import visibility_ratio
from .types import BlazePoseVideoResult, LandmarkFrame


@dataclass
class PoseQualityReport:
    """Summary of detection coverage, confidence, and temporal stability."""

    sampled_frames: int
    detected_frames: int
    detection_coverage: float
    mean_visibility: float
    mean_landmark_count: float
    temporal_jitter: float
    quality_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


class BlazePoseQualityEvaluator:
    """Evaluate whether BlazePose output is suitable for downstream comparison."""

    def __init__(self, visibility_threshold: float = 0.5):
        self.visibility_threshold = visibility_threshold

    def frame_visibility(self, frame: LandmarkFrame) -> float:
        return visibility_ratio(frame, self.visibility_threshold)

    def temporal_jitter(self, frames: Iterable[LandmarkFrame]) -> float:
        """Estimate frame-to-frame coordinate instability over shared landmarks."""
        frames = list(frames)
        differences: List[float] = []
        for previous, current in zip(frames, frames[1:]):
            shared = set(previous.landmarks).intersection(current.landmarks)
            for index in shared:
                first = previous.landmarks[index]
                second = current.landmarks[index]
                if not first.is_visible(self.visibility_threshold):
                    continue
                if not second.is_visible(self.visibility_threshold):
                    continue
                differences.append(abs(first.x - second.x) + abs(first.y - second.y))
        return float(np.mean(differences)) if differences else 0.0

    def evaluate(self, result: BlazePoseVideoResult) -> PoseQualityReport:
        sampled_frames = len(result.frames)
        detected_frames = result.detected_frame_count()
        coverage = detected_frames / sampled_frames if sampled_frames else 0.0
        visibility = [self.frame_visibility(frame) for frame in result.frames if frame.landmarks]
        landmark_counts = [len(frame.landmarks) for frame in result.frames if frame.landmarks]
        mean_visibility = float(np.mean(visibility)) if visibility else 0.0
        mean_landmark_count = float(np.mean(landmark_counts)) if landmark_counts else 0.0
        jitter = self.temporal_jitter(result.frames)
        stability = float(np.exp(-10.0 * jitter))
        quality_score = float(
            np.clip(
                100.0 * (0.45 * coverage + 0.35 * mean_visibility + 0.20 * stability),
                0.0,
                100.0,
            )
        )
        return PoseQualityReport(
            sampled_frames=sampled_frames,
            detected_frames=detected_frames,
            detection_coverage=coverage,
            mean_visibility=mean_visibility,
            mean_landmark_count=mean_landmark_count,
            temporal_jitter=jitter,
            quality_score=quality_score,
        )

    def passes(self, report: PoseQualityReport, minimum_score: float = 60.0) -> bool:
        return report.quality_score >= minimum_score
