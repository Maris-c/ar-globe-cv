"""
feature_matcher.py — ORB feature extraction and BFMatcher matching with
Lowe's ratio test.

This module is intentionally stateless — callers manage the reference
descriptors themselves.  See WebcamApp for the stateful wrapper.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import CFG


# ---------------------------------------------------------------------------
# ORB detector + BFMatcher (module-level singletons, created once)
# ---------------------------------------------------------------------------

def _build_orb(n_features: int) -> cv2.ORB:
    return cv2.ORB_create(nfeatures=n_features)


def _build_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_compute(
    image_gray: np.ndarray,
    orb: cv2.ORB,
) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
    """Run ORB detect+compute on a grayscale image.

    Returns
    -------
    (keypoints, descriptors)
        descriptors is ``None`` if no keypoints were found.
    """
    kps, descs = orb.detectAndCompute(image_gray, None)
    return kps, descs


def match_features(
    desc_ref: np.ndarray,
    desc_query: np.ndarray,
    matcher: cv2.BFMatcher,
    lowe_ratio: float,
) -> list[cv2.DMatch]:
    """Run kNN match (k=2) and apply Lowe's ratio test.

    Parameters
    ----------
    desc_ref:
        Descriptors from the reference (map) image.
    desc_query:
        Descriptors from the current webcam frame.
    matcher:
        Pre-built BFMatcher instance.
    lowe_ratio:
        Distance ratio threshold for Lowe's test (0.0–1.0).
        Lower values → stricter, fewer matches.

    Returns
    -------
    List of good DMatch objects that passed the ratio test.
    """
    if desc_ref is None or desc_query is None:
        return []
    if len(desc_ref) < 2 or len(desc_query) < 2:
        return []

    try:
        raw_matches = matcher.knnMatch(desc_ref, desc_query, k=2)
    except cv2.error:
        return []

    good: list[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)
    return good


class FeatureMatcher:
    """Stateful ORB + BFMatcher wrapper that caches reference descriptors.

    Parameters
    ----------
    n_features:
        Number of ORB features to detect.
    lowe_ratio:
        Ratio test threshold.
    """

    def __init__(
        self,
        n_features: int = 1500,
        lowe_ratio: float = 0.75,
    ) -> None:
        self._orb = _build_orb(n_features)
        self._matcher = _build_matcher()
        self._lowe_ratio = lowe_ratio

        # Reference image data (set via set_reference)
        self._ref_kps: list[cv2.KeyPoint] = []
        self._ref_descs: Optional[np.ndarray] = None
        self._ref_shape: tuple[int, int] = (0, 0)  # (H, W)

    # ------------------------------------------------------------------

    def set_reference(self, ref_gray: np.ndarray) -> int:
        """Compute and cache features for *ref_gray*.

        Returns
        -------
        Number of keypoints detected in the reference image.
        Raises ``ValueError`` if no features could be found.
        """
        kps, descs = detect_and_compute(ref_gray, self._orb)
        if descs is None or len(kps) == 0:
            raise ValueError(
                "ORB found no keypoints in the reference image.  "
                "The image may be too small, too uniform, or blurred."
            )
        self._ref_kps = kps
        self._ref_descs = descs
        self._ref_shape = ref_gray.shape[:2]
        print(f"[FeatureMatcher] Reference set: {len(kps)} keypoints detected.")
        return len(kps)

    def match(
        self,
        query_gray: np.ndarray,
    ) -> tuple[list[cv2.DMatch], list[cv2.KeyPoint]]:
        """Match query frame against the stored reference.

        Returns
        -------
        (good_matches, query_keypoints)
        """
        q_kps, q_descs = detect_and_compute(query_gray, self._orb)
        if q_descs is None:
            return [], q_kps

        good = match_features(
            self._ref_descs, q_descs, self._matcher, self._lowe_ratio
        )
        return good, q_kps

    @property
    def ref_keypoints(self) -> list[cv2.KeyPoint]:
        return self._ref_kps

    @property
    def ref_shape(self) -> tuple[int, int]:
        return self._ref_shape  # (H, W)

    @property
    def is_ready(self) -> bool:
        return self._ref_descs is not None
