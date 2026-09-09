"""YOLO11 Pose trajectory extraction utilities."""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINTS = [
    (5, "left_shoulder", "Left shoulder"),
    (6, "right_shoulder", "Right shoulder"),
    (9, "left_wrist", "Left wrist"),
    (10, "right_wrist", "Right wrist"),
    (11, "left_hip", "Left hip"),
    (12, "right_hip", "Right hip"),
    (15, "left_ankle", "Left ankle"),
]


def get_every_pose_sequence(video_path, confidence_threshold=0.3, frame_skip=10):
    """Extract normalized pose keypoints from sampled video frames."""
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")

    model_path = os.getenv("YOLO_POSE_MODEL", "yolo11n-pose.pt")
    print(f"Loading pose model: {model_path}")
    model = YOLO(model_path)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    print(
        f"Video metadata: frames={frame_count}, fps={fps:.2f}, "
        f"duration={duration:.2f}s, resolution={width}x{height}"
    )

    sampled_frames = []
    frame_index = 0
    processed_frames = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            results = model(frame, conf=confidence_threshold, verbose=False)
            people = []
            for result in results:
                if result.keypoints is None or len(result.keypoints.data) == 0:
                    continue
                keypoints = result.keypoints.data[0].cpu().numpy()
                person = {}
                for coco_index, key, _ in JOINTS:
                    if coco_index >= len(keypoints):
                        continue
                    x, y, confidence = keypoints[coco_index]
                    person[key] = {
                        "x": float(x) / width if width else 0.0,
                        "y": 1.0 - float(y) / height if height else 0.0,
                        "x_raw": float(x),
                        "y_raw": float(y),
                        "confidence": float(confidence),
                    }
                people.append(person)

            sampled_frames.append(
                {
                    "frame_idx": frame_index,
                    "timestamp": frame_index / fps if fps > 0 else 0.0,
                    "poses": people,
                }
            )
            processed_frames += 1
            frame_index += 1
            if processed_frames % 10 == 0:
                print(
                    f"Processed {processed_frames} sampled frames "
                    f"through source frame {frame_index}/{frame_count}"
                )
    finally:
        capture.release()

    reduction = (
        100.0 * (frame_count - processed_frames) / frame_count
        if frame_count
        else 0.0
    )
    print(
        f"Pose extraction completed: source_frames={frame_count}, "
        f"processed_frames={processed_frames}, reduction={reduction:.1f}%"
    )
    return {
        "video_info": {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "processed_frames": processed_frames,
            "frame_skip": frame_skip,
            "original_frames": frame_count,
            "frame_reduction": f"{reduction:.1f}%",
        },
        "poses": sampled_frames,
    }


def extract_main_person_trajectory(pose_sequence_data):
    """Convert sampled detections into one trajectory per selected joint."""
    trajectory = {
        index: {
            "name": key,
            "display_name": display_name,
            "x": [],
            "y": [],
            "x_raw": [],
            "y_raw": [],
            "conf": [],
            "frame_idx": [],
        }
        for index, (_, key, display_name) in enumerate(JOINTS)
    }

    for frame_data in pose_sequence_data["poses"]:
        frame_index = frame_data["frame_idx"]
        main_person = frame_data["poses"][0] if frame_data["poses"] else {}
        for index, (_, key, _) in enumerate(JOINTS):
            point = main_person.get(key)
            trajectory[index]["x"].append(point["x"] if point else None)
            trajectory[index]["y"].append(point["y"] if point else None)
            trajectory[index]["x_raw"].append(point["x_raw"] if point else None)
            trajectory[index]["y_raw"].append(point["y_raw"] if point else None)
            trajectory[index]["conf"].append(point["confidence"] if point else 0.0)
            trajectory[index]["frame_idx"].append(frame_index)

    return {
        "video_info": pose_sequence_data["video_info"],
        "trajectory": trajectory,
    }


def plot_pose_trajectory(trajectory_data, output_path=None, show_plot=False):
    """Plot normalized coordinates, confidence, and XY trajectories."""
    trajectory = trajectory_data["trajectory"]
    video_info = trajectory_data["video_info"]
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    figure.suptitle(f"Pose trajectory: {Path(video_info['path']).name}")

    for index, joint in trajectory.items():
        color = colors[int(index) % len(colors)]
        frames = joint["frame_idx"]
        x_values = [np.nan if value is None else value for value in joint["x"]]
        y_values = [np.nan if value is None else value for value in joint["y"]]
        axes[0, 0].plot(frames, x_values, label=joint["display_name"], color=color)
        axes[0, 1].plot(frames, y_values, label=joint["display_name"], color=color)
        axes[1, 0].plot(frames, joint["conf"], label=joint["display_name"], color=color)
        valid_xy = [
            (x, y) for x, y in zip(joint["x"], joint["y"])
            if x is not None and y is not None
        ]
        if valid_xy:
            x_track, y_track = zip(*valid_xy)
            axes[1, 1].plot(x_track, y_track, label=joint["display_name"], color=color)

    plot_settings = [
        (axes[0, 0], "Normalized X trajectories", "Frame", "Normalized X"),
        (axes[0, 1], "Normalized Y trajectories", "Frame", "Normalized Y"),
        (axes[1, 0], "Keypoint confidence", "Frame", "Confidence"),
        (axes[1, 1], "Normalized XY trajectories", "Normalized X", "Normalized Y"),
    ]
    for axis, title, xlabel, ylabel in plot_settings:
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=130, bbox_inches="tight")
        print(f"Trajectory plot saved to: {output_path}")
    if show_plot:
        plt.show()
    plt.close(figure)
    return figure


def analyze_video_pose(
    video_path,
    output_dir=None,
    confidence_threshold=0.3,
    frame_skip=10,
):
    """Run pose extraction, serialize trajectories, and create a plot."""
    print("Starting YOLO11 pose analysis")
    sequence = get_every_pose_sequence(video_path, confidence_threshold, frame_skip)
    trajectory_data = extract_main_person_trajectory(sequence)
    destination = Path(output_dir) if output_dir else Path(video_path).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem
    json_path = destination / f"{video_stem}_trajectory.json"
    plot_path = destination / f"{video_stem}_trajectory_plot.png"
    json_path.write_text(
        json.dumps(trajectory_data, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Trajectory data saved to: {json_path}")
    plot_pose_trajectory(trajectory_data, output_path=plot_path)
    print("Pose analysis completed")
    return {
        "json_path": str(json_path),
        "trajectory_plot": str(plot_path),
        "trajectory_data": trajectory_data,
        "processed_frames": sequence["video_info"]["processed_frames"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract YOLO11 pose trajectories")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--frame-skip", type=int, default=10)
    args = parser.parse_args()
    analyze_video_pose(
        args.video,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()
