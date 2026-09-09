"""End-to-end video pipeline built on MediaPipe BlazePose."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

from .config import BlazePoseConfig
from .detector import BlazePoseDetector
from .features import BlazePoseFeatureExtractor
from .io import export_landmarks_csv, save_trajectory, save_video_result
from .quality import BlazePoseQualityEvaluator
from .trajectory import BlazePoseTrajectoryBuilder
from .types import BlazePoseVideoResult


@dataclass
class BlazePosePipelineOutput:
    """Paths and summaries produced by one pipeline run."""

    raw_landmarks_json: str
    trajectory_json: str
    landmarks_csv: str
    annotated_video: Optional[str]
    quality_report: Dict
    sequence_features: Dict

    def to_dict(self) -> Dict:
        return {
            "raw_landmarks_json": self.raw_landmarks_json,
            "trajectory_json": self.trajectory_json,
            "landmarks_csv": self.landmarks_csv,
            "annotated_video": self.annotated_video,
            "quality_report": self.quality_report,
            "sequence_features": self.sequence_features,
        }


class BlazePoseVideoProcessor:
    """Read a video, sample frames, and run BlazePose detection."""

    def __init__(self, detector: BlazePoseDetector, config: BlazePoseConfig):
        self.detector = detector
        self.config = config

    def process(self, video_path, annotated_video_path=None) -> BlazePoseVideoResult:
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Video file does not exist: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if annotated_video_path:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(annotated_video_path),
                codec,
                fps if fps > 0 else 30.0,
                (width, height),
            )

        frames = []
        frame_index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if frame_index % self.config.frame_skip == 0:
                    timestamp = frame_index / fps if fps > 0 else 0.0
                    detected = self.detector.detect_bgr(image, frame_index, timestamp)
                    frames.append(detected)
                    if writer is not None:
                        writer.write(self.detector.draw_landmarks(image, detected))
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        processed = len(frames)
        video_info = {
            "path": str(source),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": frame_count / fps if fps > 0 else 0.0,
            "processed_frames": processed,
            "frame_skip": self.config.frame_skip,
            "detector": "MediaPipe BlazePose",
        }
        return BlazePoseVideoResult(video_info=video_info, frames=frames)


class BlazePosePipeline:
    """Coordinate detection, trajectory export, features, and quality evaluation."""

    def __init__(self, config: Optional[BlazePoseConfig] = None):
        self.config = config or BlazePoseConfig()
        self.config.validate()

    def run(self, video_path, output_dir="blazepose_output", annotate=False) -> BlazePosePipelineOutput:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        annotated_path = output_dir / f"{stem}_blazepose_annotated.mp4" if annotate else None

        with BlazePoseDetector(self.config) as detector:
            result = BlazePoseVideoProcessor(detector, self.config).process(
                video_path,
                annotated_video_path=annotated_path,
            )

        builder = BlazePoseTrajectoryBuilder.from_video_result(
            result,
            visibility_threshold=self.config.visibility_threshold,
        )
        trajectory = builder.build_smoothed(result.video_info, window_size=5)
        quality = BlazePoseQualityEvaluator(self.config.visibility_threshold).evaluate(result)
        feature_extractor = BlazePoseFeatureExtractor(self.config.visibility_threshold)
        sequence_features = {
            "left_wrist_motion": feature_extractor.sequence_motion(result.frames, 15),
            "right_wrist_motion": feature_extractor.sequence_motion(result.frames, 16),
            "left_ankle_motion": feature_extractor.sequence_motion(result.frames, 27),
            "right_ankle_motion": feature_extractor.sequence_motion(result.frames, 28),
            "frame_features": [
                feature_extractor.extract_frame_features(frame) for frame in result.frames
            ],
        }

        raw_path = save_video_result(result, output_dir / f"{stem}_blazepose_landmarks.json")
        trajectory_path = save_trajectory(
            trajectory,
            output_dir / f"{stem}_blazepose_trajectory.json",
        )
        csv_path = export_landmarks_csv(
            result.frames,
            output_dir / f"{stem}_blazepose_landmarks.csv",
        )
        return BlazePosePipelineOutput(
            raw_landmarks_json=raw_path,
            trajectory_json=trajectory_path,
            landmarks_csv=csv_path,
            annotated_video=str(annotated_path) if annotated_path else None,
            quality_report=quality.to_dict(),
            sequence_features=sequence_features,
        )
