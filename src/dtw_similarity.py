"""Dynamic Time Warping utilities for pose trajectory comparison."""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
from dtaidistance import dtw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_JOINTS = [
    "Left shoulder",
    "Right shoulder",
    "Left wrist",
    "Right wrist",
    "Left hip",
    "Right hip",
    "Left ankle",
]


def load_trajectory_data(json_path):
    """Load one trajectory JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        print(f"Loaded trajectory data: {json_path}")
        return data
    except Exception as exc:
        raise RuntimeError(f"Failed to load trajectory data {json_path}: {exc}") from exc


def extract_valid_trajectory(joint_data, confidence_threshold=0.3):
    """Return valid X and Y samples after confidence filtering."""
    x_values = joint_data.get("x", [])
    y_values = joint_data.get("y", [])
    confidence = joint_data.get("conf", [1.0] * len(x_values))
    valid_x = []
    valid_y = []
    for x_value, y_value, score in zip(x_values, y_values, confidence):
        if x_value is None or y_value is None or score < confidence_threshold:
            continue
        valid_x.append(float(x_value))
        valid_y.append(float(y_value))
    return valid_x, valid_y


def interpolate_trajectory(trajectory, target_length):
    """Resample a trajectory to a target length with linear interpolation."""
    if target_length <= 0 or not trajectory:
        return []
    if len(trajectory) == target_length:
        return list(trajectory)
    if len(trajectory) == 1:
        return [float(trajectory[0])] * target_length
    source_axis = np.linspace(0.0, 1.0, len(trajectory))
    target_axis = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_axis, source_axis, trajectory).tolist()


def normalize_trajectory_length(first, second, method="interpolate"):
    """Normalize two trajectories to equal lengths."""
    if not first or not second:
        return [], []
    if method == "interpolate":
        target_length = max(len(first), len(second))
        return (
            interpolate_trajectory(first, target_length),
            interpolate_trajectory(second, target_length),
        )
    if method == "truncate":
        target_length = min(len(first), len(second))
        return first[:target_length], second[:target_length]
    if method == "pad":
        target_length = max(len(first), len(second))
        first_padded = list(first) + [first[-1]] * (target_length - len(first))
        second_padded = list(second) + [second[-1]] * (target_length - len(second))
        return first_padded, second_padded
    raise ValueError(f"Unsupported normalization method: {method}")


def smooth_trajectory(trajectory, window_size=3):
    """Apply a centered moving average."""
    if window_size <= 1 or len(trajectory) < window_size:
        return list(trajectory)
    radius = window_size // 2
    return [
        float(np.mean(trajectory[max(0, index - radius): index + radius + 1]))
        for index in range(len(trajectory))
    ]


def compute_trajectory_dtw(first, second):
    """Calculate DTW distance; lower values indicate closer trajectories."""
    if not first or not second:
        return math.inf
    return float(dtw.distance(np.asarray(first), np.asarray(second)))


def dtw_distance_to_similarity(distance, max_distance=1.0):
    """Convert a nonnegative DTW distance to a similarity in [0, 1]."""
    if not math.isfinite(distance):
        return 0.0
    return float(np.clip(np.exp(-distance / max_distance), 0.0, 1.0))


def compute_joint_similarity(
    test_joint,
    reference_joint,
    normalize_method="interpolate",
    smooth_window=3,
):
    """Compare one joint using normalized Y-coordinate trajectories."""
    test_x, test_y = extract_valid_trajectory(test_joint)
    reference_x, reference_y = extract_valid_trajectory(reference_joint)
    original_lengths = [len(test_y), len(reference_y)]
    if not test_y or not reference_y:
        return {
            "y_distance": math.inf,
            "combined_distance": math.inf,
            "y_similarity": 0.0,
            "combined_similarity": 0.0,
            "valid_points_1": len(test_y),
            "valid_points_2": len(reference_y),
            "length_ratio": 0.0,
            "preprocessing": {
                "original_lengths": original_lengths,
                "normalize_method": normalize_method,
                "smooth_window": smooth_window,
            },
            "trajectories": {
                "x1": test_x,
                "y1": test_y,
                "x2": reference_x,
                "y2": reference_y,
            },
        }

    test_y = smooth_trajectory(test_y, min(smooth_window, 3))
    reference_y = smooth_trajectory(reference_y, min(smooth_window, 3))
    test_x = smooth_trajectory(test_x, min(smooth_window, 3))
    reference_x = smooth_trajectory(reference_x, min(smooth_window, 3))
    if normalize_method == "interpolate":
        target_length = len(test_y)
        test_y_normalized = test_y
        reference_y_normalized = interpolate_trajectory(reference_y, target_length)
        test_x_normalized = test_x
        reference_x_normalized = interpolate_trajectory(reference_x, target_length)
    else:
        test_y_normalized, reference_y_normalized = normalize_trajectory_length(
            test_y, reference_y, normalize_method
        )
        test_x_normalized, reference_x_normalized = normalize_trajectory_length(
            test_x, reference_x, normalize_method
        )

    distance = compute_trajectory_dtw(test_y_normalized, reference_y_normalized)
    length_ratio = min(original_lengths) / max(original_lengths)
    length_penalty = 0.5 + 0.5 * length_ratio
    similarity = dtw_distance_to_similarity(distance) * length_penalty
    return {
        "y_distance": distance,
        "combined_distance": distance,
        "y_similarity": similarity,
        "combined_similarity": similarity,
        "valid_points_1": original_lengths[0],
        "valid_points_2": original_lengths[1],
        "length_ratio": length_ratio,
        "length_penalty": length_penalty,
        "preprocessing": {
            "original_lengths": original_lengths,
            "normalized_lengths": [len(test_y_normalized), len(reference_y_normalized)],
            "normalize_method": normalize_method,
            "smooth_window": smooth_window,
            "calculation_mode": "Y_COORDINATE_ONLY",
        },
        "trajectories": {
            "x1": test_x,
            "y1": test_y,
            "x2": reference_x,
            "y2": reference_y,
            "x1_norm": test_x_normalized,
            "y1_norm": test_y_normalized,
            "x2_norm": reference_x_normalized,
            "y2_norm": reference_y_normalized,
        },
    }


def _compute_joint(task):
    joint_key, test_joint, reference_joint, normalize_method, smooth_window = task
    display_name = test_joint.get("display_name", test_joint.get("name", joint_key))
    print(f"Computing joint similarity: {display_name}")
    result = compute_joint_similarity(
        test_joint,
        reference_joint,
        normalize_method=normalize_method,
        smooth_window=smooth_window,
    )
    return display_name, result


def visualize_dtw_matching(joint_name, joint_result, output_dir):
    """Create a compact before-and-after trajectory comparison plot."""
    trajectories = joint_result.get("trajectories", {})
    if not trajectories.get("y1") or not trajectories.get("y2"):
        print(f"Skipping visualization for {joint_name}: no valid trajectory")
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(trajectories["y1"], label="Test trajectory")
    axes[0].plot(trajectories["y2"], label="Reference trajectory")
    axes[0].set_title(f"{joint_name}: original Y trajectories")
    axes[1].plot(trajectories.get("y1_norm", []), label="Normalized test")
    axes[1].plot(trajectories.get("y2_norm", []), label="Normalized reference")
    axes[1].set_title(f"{joint_name}: normalized Y trajectories")
    for axis in axes:
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Normalized Y")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"DTW similarity={joint_result['combined_similarity']:.3f}, "
        f"distance={joint_result['combined_distance']:.3f}"
    )
    figure.tight_layout()
    safe_name = joint_name.lower().replace(" ", "_")
    output_path = Path(output_dir) / f"{safe_name}_dtw_analysis.png"
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"DTW plot saved to: {output_path}")
    return str(output_path)


def compare_trajectories(
    json_path1,
    json_path2,
    output_dir="dtw_output",
    normalize_method="interpolate",
    smooth_window=3,
):
    """Compare a test trajectory JSON file with a reference trajectory JSON file."""
    started = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    test_data = load_trajectory_data(json_path1)
    reference_data = load_trajectory_data(json_path2)
    test_trajectory = test_data["trajectory"]
    reference_trajectory = reference_data["trajectory"]

    tasks = []
    for joint_key, test_joint in test_trajectory.items():
        if joint_key not in reference_trajectory:
            continue
        display_name = test_joint.get("display_name", test_joint.get("name", joint_key))
        if display_name in TARGET_JOINTS:
            tasks.append(
                (
                    joint_key,
                    test_joint,
                    reference_trajectory[joint_key],
                    normalize_method,
                    smooth_window,
                )
            )

    joint_results = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(tasks)))) as executor:
        futures = [executor.submit(_compute_joint, task) for task in tasks]
        for future in as_completed(futures):
            joint_name, result = future.result()
            joint_results[joint_name] = result

    similarities = [
        result["combined_similarity"]
        for result in joint_results.values()
        if math.isfinite(result["combined_distance"])
    ]
    for joint_name, result in joint_results.items():
        visualize_dtw_matching(joint_name, result, output_path)

    average_similarity = float(np.mean(similarities)) if similarities else 0.0
    standard_deviation = float(np.std(similarities)) if similarities else 0.0
    report = {
        "video_info": {
            "video1": test_data.get("video_info", {}),
            "video2": reference_data.get("video_info", {}),
        },
        "joints": joint_results,
        "summary": {
            "average_similarity": average_similarity,
            "std_similarity": standard_deviation,
            "valid_joints": len(similarities),
            "total_joints": len(joint_results),
            "calculation_mode": "ALL_JOINTS_Y_COORDINATE_ONLY",
            "joints_used": list(joint_results),
            "preprocessing_settings": {
                "normalize_method": normalize_method,
                "smooth_window": smooth_window,
            },
            "elapsed_seconds": time.time() - started,
        },
    }
    report_path = output_path / "similarity_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"DTW comparison completed: average_similarity={average_similarity:.3f}, "
        f"valid_joints={len(similarities)}, report={report_path}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare two pose trajectories with DTW")
    parser.add_argument("json1", help="Test trajectory JSON path")
    parser.add_argument("json2", help="Reference trajectory JSON path")
    parser.add_argument("--output", default="dtw_output")
    parser.add_argument(
        "--normalize",
        default="interpolate",
        choices=["interpolate", "truncate", "pad"],
    )
    parser.add_argument("--smooth", type=int, default=3)
    args = parser.parse_args()
    compare_trajectories(
        args.json1,
        args.json2,
        output_dir=args.output,
        normalize_method=args.normalize,
        smooth_window=args.smooth,
    )


if __name__ == "__main__":
    main()
