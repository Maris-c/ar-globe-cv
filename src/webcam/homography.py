"""
homography.py — Homography estimation with robust false-positive rejection.

DESIGN NOTES
------------
`cv2.findHomography` with RANSAC will almost always produce a matrix —
even for random matches.  A valid detection therefore requires ALL of:

  1. Enough good matches (>= min_matches).
  2. Enough RANSAC inliers (>= min_inliers).
  3. Inlier ratio >= min_inlier_ratio.
  4. The projected polygon must be convex, non-degenerate (finite area),
     and mostly inside the frame.
  5. The homography determinant must not be near-zero (degenerate transform).
  6. Aspect ratio of the projected quad must be plausible.

None of these alone guarantees a correct detection; together they
substantially reduce false positives.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _polygon_area(pts: np.ndarray) -> float:
    """Compute area of a (N,2) polygon using the shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _is_convex(pts: np.ndarray) -> bool:
    """Return True if the four corners form a convex quadrilateral.

    Uses both the convexHull point-count check AND a winding-order cross-product
    check so that self-intersecting (bowtie) quads — whose four extreme points
    ARE a convex hull — are also correctly rejected.
    """
    if len(pts) != 4:
        return False
    # 1. Hull must span all four vertices.
    hull = cv2.convexHull(pts.astype(np.float32), returnPoints=True)
    if hull.shape[0] != 4:
        return False
    # 2. Cross products of consecutive edges must all have the same sign
    #    (consistent winding order = true convex polygon).
    signs = []
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cross = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if abs(cross) > 1e-6:
            signs.append(cross > 0)
    if not signs:
        return False
    return all(s == signs[0] for s in signs)


def _aspect_ratio(pts: np.ndarray) -> float:
    """Estimate width/height aspect ratio of the quad (min/max edge pair)."""
    a, b, c, d = pts
    w1 = math.hypot(*(b - a))
    w2 = math.hypot(*(d - c))
    h1 = math.hypot(*(c - b))
    h2 = math.hypot(*(a - d))
    w = (w1 + w2) / 2.0
    h = (h1 + h2) / 2.0
    if h < 1e-3:
        return 0.0
    return w / h


def _mostly_in_frame(pts: np.ndarray, frame_w: int, frame_h: int, margin: float = 0.5) -> bool:
    """Return True if at least (1-margin) fraction of corners are inside the frame."""
    inside = np.sum((pts[:, 0] >= 0) & (pts[:, 0] <= frame_w) &
                    (pts[:, 1] >= 0) & (pts[:, 1] <= frame_h))
    return inside >= (1.0 - margin) * 4


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def estimate_homography(
    ref_kps: list[cv2.KeyPoint],
    query_kps: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    min_matches: int = 15,
    ransac_reproj_threshold: float = 5.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate a homography from matched keypoints.

    Parameters
    ----------
    ref_kps, query_kps:
        Keypoints from the reference and query images respectively.
    good_matches:
        Matches that passed Lowe's ratio test.
    min_matches:
        Hard lower bound on number of matches needed before RANSAC runs.
    ransac_reproj_threshold:
        RANSAC reprojection error threshold in pixels.

    Returns
    -------
    (H, mask)
        H is the 3×3 homography matrix; mask is the RANSAC inlier mask (uint8).
        Returns (None, None) if not enough matches or RANSAC fails.
    """
    if len(good_matches) < min_matches:
        return None, None

    src_pts = np.float32([ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([query_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    return H, mask


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def validate_homography(
    H: np.ndarray,
    mask: np.ndarray,
    ref_h: int,
    ref_w: int,
    frame_h: int,
    frame_w: int,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.25,
    min_area_fraction: float = 0.001,
    max_aspect_ratio: float = 20.0,
) -> tuple[bool, str, Optional[np.ndarray]]:
    """Validate a homography against multiple rejection criteria.

    Parameters
    ----------
    H:
        3×3 homography matrix.
    mask:
        RANSAC inlier mask returned by findHomography.
    ref_h, ref_w:
        Reference image dimensions.
    frame_h, frame_w:
        Query frame dimensions.
    min_inliers:
        Minimum number of RANSAC inliers required.
    min_inlier_ratio:
        Minimum fraction of good matches that must be inliers.
    min_area_fraction:
        Minimum projected area as a fraction of the frame area.
    max_aspect_ratio:
        Maximum allowed aspect ratio of the projected quad.

    Returns
    -------
    (is_valid, reason, projected_corners)
        projected_corners is shape (4,1,2) float32 if valid, else None.
    """
    if H is None or mask is None:
        return False, "No homography computed", None

    inliers = int(mask.sum())
    total = len(mask)
    ratio = inliers / total if total > 0 else 0.0

    if inliers < min_inliers:
        return False, f"Too few inliers: {inliers} < {min_inliers}", None

    if ratio < min_inlier_ratio:
        return False, f"Low inlier ratio: {ratio:.2f} < {min_inlier_ratio:.2f}", None

    # Check homography is not degenerate (near-zero determinant).
    det = np.linalg.det(H[:2, :2])
    if abs(det) < 1e-6:
        return False, f"Degenerate homography (det={det:.2e})", None

    # Project reference corners into frame space.
    corners_ref = np.float32([
        [0,         0],
        [ref_w,     0],
        [ref_w, ref_h],
        [0,     ref_h],
    ]).reshape(-1, 1, 2)

    try:
        corners_frame = cv2.perspectiveTransform(corners_ref, H)
    except cv2.error as exc:
        return False, f"perspectiveTransform failed: {exc}", None

    pts = corners_frame.reshape(4, 2)

    # Convexity check.
    if not _is_convex(pts):
        return False, "Projected quad is not convex", None

    # Area sanity check.
    area = _polygon_area(pts)
    frame_area = frame_h * frame_w
    if area < min_area_fraction * frame_area:
        return False, f"Projected area too small: {area:.0f}px² < {min_area_fraction*frame_area:.0f}px²", None

    # Aspect ratio sanity check (ref AR as reference).
    proj_ar = _aspect_ratio(pts)
    ref_ar = ref_w / ref_h if ref_h > 0 else 1.0
    ar_deviation = max(proj_ar / ref_ar, ref_ar / proj_ar) if proj_ar > 0 else 999
    if ar_deviation > max_aspect_ratio:
        return False, f"Extreme aspect ratio deviation: {ar_deviation:.1f}x", None

    # At least half of corners must be visible in frame.
    if not _mostly_in_frame(pts, frame_w, frame_h):
        return False, "Projected quad mostly outside frame", None

    return True, f"OK ({inliers}/{total} inliers, ratio={ratio:.2f})", corners_frame


# ---------------------------------------------------------------------------
# Point mapping
# ---------------------------------------------------------------------------

def map_frame_point_to_ref(
    frame_x: float,
    frame_y: float,
    H: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Map a click point in the webcam frame back to reference image coordinates.

    Uses the inverse homography  H⁻¹ · [frame_x, frame_y, 1]ᵀ.

    Returns
    -------
    (ref_x, ref_y) or None if H is not invertible.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    pt = np.float32([[[frame_x, frame_y]]])
    ref_pt = cv2.perspectiveTransform(pt, H_inv)
    rx, ry = float(ref_pt[0][0][0]), float(ref_pt[0][0][1])
    return rx, ry
