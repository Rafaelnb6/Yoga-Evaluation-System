"""Unit tests for BlazePose trajectory conversion."""

from src.blazepose import (
    BlazePoseLandmark,
    BlazePoseTrajectoryBuilder,
    LandmarkFrame,
    LandmarkPoint,
)


def test_builder_exports_selected_landmarks():
    frame = LandmarkFrame(
        frame_index=4,
        timestamp=0.2,
        landmarks={
            int(BlazePoseLandmark.LEFT_SHOULDER): LandmarkPoint(0.2, 0.8),
            int(BlazePoseLandmark.RIGHT_SHOULDER): LandmarkPoint(0.8, 0.8),
        },
    )
    builder = BlazePoseTrajectoryBuilder()
    builder.append(frame)
    result = builder.build({"fps": 20.0})
    assert len(result["trajectory"]) == 7
    assert result["trajectory"][0]["display_name"] == "Left shoulder"
    assert result["trajectory"][0]["frame_idx"] == [4]
    assert result["trajectory"][0]["x"] == [0.2]


def test_builder_filters_low_visibility_points():
    frame = LandmarkFrame(
        frame_index=0,
        timestamp=0.0,
        landmarks={
            int(BlazePoseLandmark.LEFT_WRIST): LandmarkPoint(
                0.4,
                0.5,
                visibility=0.1,
            )
        },
    )
    builder = BlazePoseTrajectoryBuilder(visibility_threshold=0.5)
    builder.append(frame)
    result = builder.build({})
    assert result["trajectory"][2]["x"] == [None]
    assert result["trajectory"][2]["conf"] == [0.1]
