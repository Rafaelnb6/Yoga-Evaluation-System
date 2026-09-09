"""Command-line entry point for the paper's released core pipeline."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.deepseek_advisor import get_deepseek_advice
from src.dtw_similarity import compare_trajectories
from src.pose_extraction import analyze_video_pose


def similarity_to_score(similarity: float) -> float:
    """Map [0, 1] similarity to the system's deterministic [60, 100] score."""
    return round(max(60.0, min(100.0, 60.0 + 40.0 * similarity)), 1)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="YOLOPose + DTW + DeepSeek pipeline")
    parser.add_argument("video", help="Path to the test video")
    parser.add_argument(
        "--standard-trajectory",
        default="examples/standard_trajectory.json",
        help="Reference trajectory JSON",
    )
    parser.add_argument("--output-dir", default="outputs", help="Result directory")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--no-ai", action="store_true", help="Skip the DeepSeek call")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pose_dir = output_dir / "pose"
    dtw_dir = output_dir / "dtw"
    pose_dir.mkdir(parents=True, exist_ok=True)
    dtw_dir.mkdir(parents=True, exist_ok=True)

    pose_result = analyze_video_pose(
        args.video,
        output_dir=str(pose_dir),
        confidence_threshold=args.confidence,
        frame_skip=args.frame_skip,
    )
    dtw_result = compare_trajectories(
        pose_result["json_path"],
        args.standard_trajectory,
        output_dir=str(dtw_dir),
        normalize_method="interpolate",
        smooth_window=3,
    )

    similarity = float(dtw_result["summary"]["average_similarity"])
    joint_scores = {
        name: float(values["combined_similarity"])
        for name, values in dtw_result["joints"].items()
    }
    result = {
        "average_similarity": similarity,
        "overall_score": similarity_to_score(similarity),
        "joint_scores": joint_scores,
        "trajectory_json": pose_result["json_path"],
        "trajectory_plot": pose_result["trajectory_plot"],
    }

    if not args.no_ai:
        try:
            result["deepseek_advice"] = get_deepseek_advice(
                joint_scores, result["overall_score"]
            )
        except RuntimeError as exc:
            result["deepseek_advice"] = None
            result["deepseek_notice"] = str(exc)

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
