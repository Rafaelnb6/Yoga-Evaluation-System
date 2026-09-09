"""Command-line entry point for the standalone BlazePose pipeline."""

import argparse
import json

from src.blazepose import BlazePoseConfig, BlazePosePipeline


def main():
    parser = argparse.ArgumentParser(description="Run standalone BlazePose analysis")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output-dir", default="blazepose_output")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    config = BlazePoseConfig(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        visibility_threshold=args.visibility_threshold,
        frame_skip=args.frame_skip,
    )
    output = BlazePosePipeline(config).run(
        args.video,
        output_dir=args.output_dir,
        annotate=args.annotate,
    )
    print(json.dumps(output.to_dict(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
