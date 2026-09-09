"""Serialization helpers for BlazePose data."""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable

from .types import BlazePoseVideoResult, LandmarkFrame


def save_video_result(result: BlazePoseVideoResult, path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return str(destination)


def load_video_result(path) -> BlazePoseVideoResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BlazePoseVideoResult.from_dict(data)


def save_trajectory(trajectory: Dict, path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(trajectory, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return str(destination)


def export_landmarks_csv(frames: Iterable[LandmarkFrame], path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp",
                "landmark_index",
                "x",
                "y",
                "z",
                "visibility",
                "presence",
            ]
        )
        for frame in frames:
            for index, point in sorted(frame.landmarks.items()):
                writer.writerow(
                    [
                        frame.frame_index,
                        frame.timestamp,
                        index,
                        point.x,
                        point.y,
                        point.z,
                        point.visibility,
                        point.presence,
                    ]
                )
    return str(destination)
