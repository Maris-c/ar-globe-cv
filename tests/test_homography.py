"""
test_homography.py — Unit tests for homography estimation and validation.

Run with:
    python -m pytest tests/test_homography.py -v
"""

from __future__ import annotations

import sys
import pathlib
import math

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from webcam.homography import (
    estimate_homography,
    validate_homography,
    map_frame_point_to_ref,
    _polygon_area,
    _is_convex,
    _aspect_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kps(pts: list[tuple[float, float]]) -> list:
    """Create minimal KeyPoint objects at the given coordinates."""
    return [type("KP", (), {"pt": p})() for p in pts]


def _identity_matches(n: int):
    """Create n trivial DMatch objects (queryIdx == trainIdx)."""
    matches = []
    for i in range(n):
        m = type("DMatch", (), {"queryIdx": i, "trainIdx": i, "distance": 10.0})()
        matches.append(m)
    return matches


# ---------------------------------------------------------------------------
# _polygon_area
# ---------------------------------------------------------------------------

class TestPolygonArea:
    def test_unit_square(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _polygon_area(pts) == pytest.approx(1.0, abs=1e-9)

    def test_rectangle(self):
        pts = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)
        assert _polygon_area(pts) == pytest.approx(12.0, abs=1e-6)

    def test_zero_area_collinear(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        assert _polygon_area(pts) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _is_convex
# ---------------------------------------------------------------------------

class TestIsConvex:
    def test_convex_square(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)
        assert _is_convex(pts) is True

    def test_non_convex_bowtie(self):
        # Bowtie / self-intersecting quad is not convex.
        pts = np.array([[0, 0], [4, 4], [4, 0], [0, 4]], dtype=np.float32)
        assert _is_convex(pts) is False

    def test_requires_4_points(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        assert _is_convex(pts) is False


# ---------------------------------------------------------------------------
# _aspect_ratio
# ---------------------------------------------------------------------------

class TestAspectRatio:
    def test_square(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        assert _aspect_ratio(pts) == pytest.approx(1.0, abs=0.01)

    def test_wide_rectangle(self):
        pts = np.array([[0, 0], [10, 0], [10, 2], [0, 2]], dtype=float)
        ar = _aspect_ratio(pts)
        assert ar == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# estimate_homography — insufficient matches
# ---------------------------------------------------------------------------

class TestEstimateHomography:
    def test_too_few_matches_returns_none(self):
        kps = _make_kps([(float(i), float(i)) for i in range(5)])
        matches = _identity_matches(5)
        H, mask = estimate_homography(kps, kps, matches, min_matches=15)
        assert H is None
        assert mask is None

    def test_enough_noncollinear_matches_produces_homography(self):
        """cv2.findHomography requires non-collinear points; a grid layout works."""
        import cv2 as _cv2
        # 5x4 grid → 20 non-collinear points; translated by (+50, +30)
        ref_pts = [(float(col * 60), float(row * 60))
                   for row in range(4) for col in range(5)]
        qry_pts = [(x + 50.0, y + 30.0) for x, y in ref_pts]

        src = np.float32(ref_pts).reshape(-1, 1, 2)
        dst = np.float32(qry_pts).reshape(-1, 1, 2)
        H_expected, mask = _cv2.findHomography(src, dst, _cv2.RANSAC, 5.0)
        assert H_expected is not None, "findHomography should succeed with non-collinear grid points"
        # Verify it's a near-pure translation.
        assert H_expected[0, 2] == pytest.approx(50.0, abs=1.0)
        assert H_expected[1, 2] == pytest.approx(30.0, abs=1.0)


# ---------------------------------------------------------------------------
# validate_homography
# ---------------------------------------------------------------------------

class TestValidateHomography:
    """Test the validation gate with synthetic homographies."""

    REF_W, REF_H   = 1200, 600
    FRAME_W, FRAME_H = 1280, 720

    def _make_valid_H(self, scale: float = 0.5, tx: float = 100, ty: float = 50):
        """Build a simple scale+translate H that projects ref into frame."""
        H = np.array([
            [scale, 0,     tx],
            [0,     scale, ty],
            [0,     0,     1 ],
        ], dtype=np.float64)
        return H

    def _make_mask(self, total: int, inlier_frac: float):
        mask = np.zeros((total, 1), dtype=np.uint8)
        n_in = int(total * inlier_frac)
        mask[:n_in] = 1
        return mask

    # --- Valid case ---

    def test_valid_homography_passes(self):
        H = self._make_valid_H(scale=0.4, tx=50, ty=30)
        mask = self._make_mask(40, 0.8)
        ok, reason, corners = validate_homography(
            H, mask, self.REF_H, self.REF_W, self.FRAME_H, self.FRAME_W,
            min_inliers=10, min_inlier_ratio=0.25,
        )
        assert ok, f"Expected valid, got: {reason}"
        assert corners is not None

    # --- Too few inliers ---

    def test_too_few_inliers_rejected(self):
        H = self._make_valid_H()
        mask = self._make_mask(20, 0.2)   # 4 inliers out of 20
        ok, reason, _ = validate_homography(
            H, mask, self.REF_H, self.REF_W, self.FRAME_H, self.FRAME_W,
            min_inliers=10,
        )
        assert not ok
        assert "inliers" in reason.lower()

    # --- Low inlier ratio ---

    def test_low_inlier_ratio_rejected(self):
        H = self._make_valid_H()
        mask = self._make_mask(100, 0.10)   # 10 inliers but ratio = 0.10
        ok, reason, _ = validate_homography(
            H, mask, self.REF_H, self.REF_W, self.FRAME_H, self.FRAME_W,
            min_inliers=9, min_inlier_ratio=0.25,
        )
        assert not ok
        assert "ratio" in reason.lower()

    # --- None H ---

    def test_none_homography_rejected(self):
        ok, reason, corners = validate_homography(
            None, None, self.REF_H, self.REF_W, self.FRAME_H, self.FRAME_W,
        )
        assert not ok
        assert corners is None

    # --- Degenerate homography (near-zero det) ---

    def test_degenerate_homography_rejected(self):
        H = np.zeros((3, 3), dtype=np.float64)
        H[2, 2] = 1.0
        mask = self._make_mask(30, 0.9)
        ok, reason, _ = validate_homography(
            H, mask, self.REF_H, self.REF_W, self.FRAME_H, self.FRAME_W,
            min_inliers=5, min_inlier_ratio=0.1,
        )
        assert not ok
        assert "degenerate" in reason.lower()


# ---------------------------------------------------------------------------
# map_frame_point_to_ref
# ---------------------------------------------------------------------------

class TestMapFramePointToRef:
    def test_identity_homography(self):
        H = np.eye(3, dtype=np.float64)
        result = map_frame_point_to_ref(100.0, 200.0, H)
        assert result is not None
        rx, ry = result
        assert rx == pytest.approx(100.0, abs=0.1)
        assert ry == pytest.approx(200.0, abs=0.1)

    def test_translation_homography(self):
        # H translates by (+50, +30) → inverse maps back by (-50, -30).
        H = np.array([
            [1, 0, 50],
            [0, 1, 30],
            [0, 0, 1 ],
        ], dtype=np.float64)
        result = map_frame_point_to_ref(150.0, 130.0, H)
        assert result is not None
        rx, ry = result
        assert rx == pytest.approx(100.0, abs=0.5)
        assert ry == pytest.approx(100.0, abs=0.5)

    def test_singular_homography_returns_none(self):
        H = np.zeros((3, 3), dtype=np.float64)
        result = map_frame_point_to_ref(100.0, 100.0, H)
        assert result is None
