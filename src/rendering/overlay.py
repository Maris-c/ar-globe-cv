"""
overlay.py — Drawing helpers for markers, labels, status bars, and info panels.

All functions operate on BGR NumPy arrays in-place (or return a copy when
explicitly documented).  They are shared by both ImageMapApp and WebcamApp.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour constants (BGR)
# ---------------------------------------------------------------------------
GREEN   = (0,  220,  80)
RED     = (0,   60, 230)
YELLOW  = (0,  220, 230)
WHITE   = (255, 255, 255)
BLACK   = (0,     0,   0)
CYAN    = (230, 200,   0)
ORANGE  = (0,  140, 255)
DARK_BG = (30,  30,  30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_marker(
    image: np.ndarray,
    x: int,
    y: int,
    index: int = 0,
    radius: int = 8,
    color: tuple[int, int, int] = GREEN,
) -> None:
    """Draw a circle + crosshair marker at *(x, y)* and annotate with *index*."""
    cv2.circle(image, (x, y), radius, color, 2)
    cv2.circle(image, (x, y), 3, color, -1)
    # crosshair arms
    arm = radius + 6
    cv2.line(image, (x - arm, y), (x + arm, y), color, 1)
    cv2.line(image, (x, y - arm), (x, y + arm), color, 1)
    # index label
    label = str(index)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = x + radius + 4, y - radius
    cv2.rectangle(image, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 4), BLACK, -1)
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_coord_label(
    image: np.ndarray,
    x: int,
    y: int,
    lat: float,
    lon: float,
    px_x: int,
    px_y: int,
) -> None:
    """Draw a coordinate tooltip near *(x, y)*."""
    lines = [
        f"Pixel: ({px_x}, {px_y})",
        f"Lat: {lat:+.4f}°",
        f"Lon: {lon:+.4f}°",
    ]
    _draw_tooltip(image, x, y, lines)


def draw_info_panel(
    image: np.ndarray,
    points: list[dict],
    status: str = "",
    shortcuts: list[str] | None = None,
) -> None:
    """Overlay a semi-transparent left-side info panel with point list and shortcuts."""
    h, w = image.shape[:2]
    panel_w = 310
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    y_cursor = 28
    _put(image, "AR Globe CV", 10, y_cursor, scale=0.75, color=CYAN, bold=True)
    y_cursor += 28

    if status:
        color = GREEN if "Detected" in status else (YELLOW if "Searching" in status else RED)
        _put(image, f"Status: {status}", 10, y_cursor, scale=0.55, color=color)
        y_cursor += 22

    cv2.line(image, (10, y_cursor), (panel_w - 10, y_cursor), (80, 80, 80), 1)
    y_cursor += 14

    # Point list
    _put(image, f"Points ({len(points)}):", 10, y_cursor, scale=0.55, color=WHITE)
    y_cursor += 20
    for i, pt in enumerate(points[-8:]):  # show last 8
        txt = f"  #{i+1}  ({pt['latitude']:+.2f}°, {pt['longitude']:+.2f}°)"
        _put(image, txt, 10, y_cursor, scale=0.48, color=GREEN)
        y_cursor += 17

    if shortcuts:
        y_cursor = max(y_cursor + 10, h - 14 * (len(shortcuts) + 2))
        cv2.line(image, (10, y_cursor), (panel_w - 10, y_cursor), (80, 80, 80), 1)
        y_cursor += 14
        _put(image, "Shortcuts:", 10, y_cursor, scale=0.50, color=YELLOW)
        y_cursor += 18
        for s in shortcuts:
            _put(image, s, 10, y_cursor, scale=0.44, color=WHITE)
            y_cursor += 16


def draw_detection_polygon(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = GREEN,
    thickness: int = 3,
) -> None:
    """Draw the four-sided bounding polygon for a detected map.

    Parameters
    ----------
    corners:
        Shape (4, 1, 2) float32 array from perspectiveTransform, or (4, 2).
    """
    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    # Corner dots
    for pt in pts.reshape(-1, 2):
        cv2.circle(image, tuple(pt), 6, ORANGE, -1)


def draw_fps(image: np.ndarray, fps: float, x: int = 10, y: int = 24) -> None:
    """Draw FPS counter on the image."""
    color = GREEN if fps >= 20 else (YELLOW if fps >= 10 else RED)
    _put(image, f"FPS: {fps:.1f}", x, y, scale=0.60, color=color, bold=True)


def draw_status_bar(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int] = WHITE,
) -> None:
    """Draw a status bar at the bottom of the image."""
    h, w = image.shape[:2]
    bar_h = 30
    cv2.rectangle(image, (0, h - bar_h), (w, h), DARK_BG, -1)
    _put(image, text, 8, h - 8, scale=0.52, color=color)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _put(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color: tuple[int, int, int] = WHITE,
    bold: bool = False,
) -> None:
    thickness = 2 if bold else 1
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_tooltip(
    image: np.ndarray,
    x: int,
    y: int,
    lines: list[str],
) -> None:
    """Draw a multi-line tooltip box anchored near *(x, y)*."""
    padding = 6
    line_h = 18
    font_scale = 0.50
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    box_w = max_w + 2 * padding
    box_h = len(lines) * line_h + 2 * padding

    h, w = image.shape[:2]
    # Prefer below-right; flip if near edge
    bx = x + 14 if x + 14 + box_w < w else x - 14 - box_w
    by = y + 14 if y + 14 + box_h < h else y - 14 - box_h

    cv2.rectangle(image, (bx, by), (bx + box_w, by + box_h), BLACK, -1)
    cv2.rectangle(image, (bx, by), (bx + box_w, by + box_h), GREEN, 1)

    for i, line in enumerate(lines):
        cy = by + padding + (i + 1) * line_h - 4
        cv2.putText(image, line, (bx + padding, cy), font, font_scale, WHITE, thickness, cv2.LINE_AA)
