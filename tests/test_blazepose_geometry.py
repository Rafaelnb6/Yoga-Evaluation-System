"""Unit tests for BlazePose geometry helpers without a model dependency."""

import math

from src.blazepose import LandmarkPoint, interpolate_missing, joint_angle, midpoint


def test_joint_angle_is_ninety_degrees():
    first = LandmarkPoint(1.0, 0.0)
    center = LandmarkPoint(0.0, 0.0)
    third = LandmarkPoint(0.0, 1.0)
    assert math.isclose(joint_angle(first, center, third), 90.0, abs_tol=1e-6)


def test_midpoint_preserves_lowest_confidence():
    first = LandmarkPoint(0.0, 0.0, visibility=0.9)
    second = LandmarkPoint(2.0, 2.0, visibility=0.6)
    result = midpoint(first, second)
    assert result.x == 1.0
    assert result.y == 1.0
    assert result.visibility == 0.6


def test_interpolate_missing_values():
    assert interpolate_missing([0.0, None, 2.0]) == [0.0, 1.0, 2.0]
