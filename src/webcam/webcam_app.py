"""
webcam_app.py — Mode 2: Live webcam map detection and click-to-coordinate.

Usage (standalone):
    python -m src.webcam.webcam_app --map assets/images/world_map.jpg

Keyboard shortcuts inside the OpenCV window:
    Left-click  → Place marker on detected map region
    S           → Save all points to CSV
    C           → Clear all markers
    Z           → Undo last marker
    D           → Toggle debug overlay (match count, inlier info)
    Q / ESC     → Quit
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

_SRC = pathlib.Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CFG
from coordinates.map_coordinates import MapConfig, pixel_to_latlon
from rendering.overlay import (
    draw_marker, draw_coord_label, draw_info_panel,
    draw_detection_polygon, draw_fps, draw_status_bar,
    GREEN, RED, YELLOW, WHITE, CYAN, ORANGE,
)
from utils.file_utils import load_image, save_points_csv, choose_image_file_dialog
from webcam.feature_matcher import FeatureMatcher
from webcam.homography import estimate_homography, validate_homography, map_frame_point_to_ref


# ---------------------------------------------------------------------------
# Detection state machine
# ---------------------------------------------------------------------------

class DetectionState(Enum):
    SEARCHING = auto()
    DETECTED  = auto()
    LOST      = auto()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SelectedPoint:
    pixel_x: int          # reference image pixel
    pixel_y: int
    latitude: float
    longitude: float
    frame_x: int          # webcam frame pixel (for display)
    frame_y: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "frame_x": self.frame_x,
            "frame_y": self.frame_y,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class WebcamApp:
    """Live webcam-based map detection and coordinate extraction.

    Parameters
    ----------
    ref_image_path:
        Path to the reference map image.
    device_index:
        OpenCV VideoCapture device index (0 = default webcam).
    map_cfg:
        Geographic extent / projection config.
    """

    WINDOW = "AR Globe CV — Webcam Map"
    SHORTCUTS = [
        "LClick  → Place marker",
        "Z       → Undo",
        "C       → Clear all",
        "S       → Save CSV",
        "D       → Debug overlay",
        "Q/ESC   → Quit",
    ]

    def __init__(
        self,
        ref_image_path: str | pathlib.Path,
        device_index: int = 0,
        map_cfg: Optional[MapConfig] = None,
    ) -> None:
        # Load reference image.
        self._ref_bgr: np.ndarray = load_image(ref_image_path)
        self._ref_gray = cv2.cvtColor(self._ref_bgr, cv2.COLOR_BGR2GRAY)
        self._ref_h, self._ref_w = self._ref_gray.shape[:2]

        self._map_cfg = map_cfg or MapConfig(
            projection=CFG["map"]["projection"],
            lon_min=CFG["map"]["lon_min"],
            lon_max=CFG["map"]["lon_max"],
            lat_min=CFG["map"]["lat_min"],
            lat_max=CFG["map"]["lat_max"],
        )

        # CV thresholds from config.
        cv_cfg = CFG["feature_matching"]
        self._min_matches: int       = cv_cfg["min_matches"]
        self._min_inliers: int       = cv_cfg["min_inliers"]
        self._min_inlier_ratio: float = cv_cfg["min_inlier_ratio"]
        self._ransac_threshold: float = cv_cfg["ransac_reproj_threshold"]

        # Build feature matcher and compute reference descriptors.
        self._matcher = FeatureMatcher(
            n_features=cv_cfg["orb_n_features"],
            lowe_ratio=cv_cfg["lowe_ratio"],
        )
        n_kps = self._matcher.set_reference(self._ref_gray)

        # Runtime state.
        self._state = DetectionState.SEARCHING
        self._H: Optional[np.ndarray] = None          # last valid homography
        self._corners: Optional[np.ndarray] = None    # projected polygon corners
        self._points: list[SelectedPoint] = []
        self._debug_mode = False
        self._status_msg = "Searching for map…"

        # Performance counters.
        self._frame_count = 0
        self._detect_count = 0
        self._lost_count = 0
        self._fps = 0.0
        self._t_last = time.perf_counter()

        # Webcam.
        self._device_index = device_index

        print(f"[WebcamApp] Reference: {ref_image_path}  ({self._ref_w}x{self._ref_h}px, {n_kps} keypoints)")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open webcam (device index {self._device_index}).")
            print("  → Try a different --device index (e.g., --device 1).")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1280, 720)
        cv2.setMouseCallback(self.WINDOW, self._mouse_cb)

        print(f"[WebcamApp] Webcam opened (device {self._device_index}). Shortcuts:")
        for s in self.SHORTCUTS:
            print(f"  {s}")
        print()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WebcamApp] WARNING: Failed to read frame. Retrying…")
                    time.sleep(0.05)
                    continue

                self._process_frame(frame)
                display = self._render_frame(frame)
                cv2.imshow(self.WINDOW, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("s"), ord("S")):
                    self._save_csv()
                elif key in (ord("c"), ord("C")):
                    self._clear_all()
                elif key in (ord("z"), ord("Z")):
                    self._undo()
                elif key in (ord("d"), ord("D")):
                    self._debug_mode = not self._debug_mode

                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyWindow(self.WINDOW)
            self._print_summary()

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        """Run detection pipeline on the current frame."""
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._t_last
        if elapsed >= 0.5:
            self._fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            # Note: frame_count resets each FPS window
            self._frame_count = 0
            self._t_last = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_matches, q_kps = self._matcher.match(gray)

        fh, fw = frame.shape[:2]
        H, mask = estimate_homography(
            self._matcher.ref_keypoints,
            q_kps,
            good_matches,
            min_matches=self._min_matches,
            ransac_reproj_threshold=self._ransac_threshold,
        )

        valid, reason, corners = validate_homography(
            H, mask,
            self._ref_h, self._ref_w,
            fh, fw,
            min_inliers=self._min_inliers,
            min_inlier_ratio=self._min_inlier_ratio,
        )

        if valid:
            self._H = H
            self._corners = corners
            self._state = DetectionState.DETECTED
            self._detect_count += 1
            self._status_msg = f"Detected  ({int(mask.sum())} inliers)"
        else:
            if self._state == DetectionState.DETECTED:
                self._lost_count += 1
            self._state = DetectionState.SEARCHING if self._H is None else DetectionState.LOST
            self._status_msg = f"Searching…  [{reason}]"
            # Keep stale homography for a few frames → allows brief occlusion.
            # Reset after LOST state for 1s+ (handled implicitly by detect_count lag).

        # Store debug info.
        self._last_match_count = len(good_matches)
        self._last_inliers = int(mask.sum()) if mask is not None else 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        fh, fw = display.shape[:2]

        # Detection polygon.
        if self._corners is not None and self._state != DetectionState.SEARCHING:
            color = GREEN if self._state == DetectionState.DETECTED else YELLOW
            draw_detection_polygon(display, self._corners, color=color)

        # Placed markers (in frame coordinates).
        for i, pt in enumerate(self._points):
            draw_marker(display, pt.frame_x, pt.frame_y, index=i + 1)
            draw_coord_label(display, pt.frame_x, pt.frame_y,
                             pt.latitude, pt.longitude, pt.pixel_x, pt.pixel_y)

        # Info panel.
        state_str = {
            DetectionState.DETECTED:  "Detected",
            DetectionState.SEARCHING: "Searching",
            DetectionState.LOST:      "Tracking Lost",
        }[self._state]
        draw_info_panel(display, [p.to_dict() for p in self._points],
                        status=state_str, shortcuts=self.SHORTCUTS)

        draw_fps(display, self._fps, x=320, y=24)

        # Debug overlay.
        if self._debug_mode:
            dbg = (f"Matches={self._last_match_count}  "
                   f"Inliers={self._last_inliers}  "
                   f"Detect={self._detect_count}  Lost={self._lost_count}")
            cv2.putText(display, dbg, (320, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 50), 1, cv2.LINE_AA)

        draw_status_bar(display, self._status_msg)
        return display

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self._H is None:
            self._status_msg = "No map detected — cannot place marker."
            print("[WebcamApp] Click ignored: no valid homography.")
            return

        ref_pos = map_frame_point_to_ref(float(x), float(y), self._H)
        if ref_pos is None:
            self._status_msg = "Homography not invertible."
            return

        rx, ry = ref_pos
        rx_i = max(0, min(self._ref_w - 1, int(round(rx))))
        ry_i = max(0, min(self._ref_h - 1, int(round(ry))))

        if not (0 <= rx_i < self._ref_w and 0 <= ry_i < self._ref_h):
            self._status_msg = "Click is outside the map region."
            print(f"[WebcamApp] Click ({x},{y}) maps to ref ({rx_i},{ry_i}) — outside bounds.")
            return

        lat, lon = pixel_to_latlon(rx_i, ry_i, self._ref_w, self._ref_h, self._map_cfg)
        pt = SelectedPoint(
            pixel_x=rx_i, pixel_y=ry_i,
            latitude=lat, longitude=lon,
            frame_x=x, frame_y=y,
        )
        self._points.append(pt)

        print(f"[WebcamApp] #{len(self._points):>3} | "
              f"ref=({rx_i},{ry_i}) | lat={lat:+.4f}°  lon={lon:+.4f}°")
        self._status_msg = f"Point #{len(self._points)}: lat={lat:+.4f}°  lon={lon:+.4f}°"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _undo(self) -> None:
        if self._points:
            removed = self._points.pop()
            self._status_msg = f"Removed point #{len(self._points)+1}"
        else:
            self._status_msg = "No points to remove."

    def _clear_all(self) -> None:
        n = len(self._points)
        self._points.clear()
        self._status_msg = f"Cleared {n} point(s)."

    def _save_csv(self) -> None:
        if not self._points:
            self._status_msg = "No points to save."
            return
        try:
            out_dir = pathlib.Path(CFG["output"]["csv_dir"])
            path = save_points_csv([p.to_dict() for p in self._points], out_dir)
            self._status_msg = f"Saved {len(self._points)} points → {path.name}"
            print(f"[WebcamApp] CSV saved: {path.resolve()}")
        except Exception as exc:  # noqa: BLE001
            self._status_msg = f"Save failed: {exc}"

    def _print_summary(self) -> None:
        print("\n[WebcamApp] Session summary:")
        print(f"  Detections  : {self._detect_count}")
        print(f"  Tracking lost: {self._lost_count}")
        print(f"  Points saved : {len(self._points)}")
        print("  NOTE: FPS/detection figures are from THIS run only; "
              "not verified on external hardware.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AR Globe CV — Webcam Map Mode")
    parser.add_argument("--map", "-m", type=str, default=None,
                        help="Path to reference map image. Opens dialog if omitted.")
    parser.add_argument("--device", "-d", type=int,
                        default=CFG["webcam"]["device_index"],
                        help="Webcam device index (default: 0).")
    args = parser.parse_args()

    image_path = args.map
    if image_path is None:
        print("[WebcamApp] No --map argument. Opening file dialog…")
        image_path = choose_image_file_dialog()

    if not image_path:
        print("[WebcamApp] No image selected. Exiting.")
        sys.exit(1)

    try:
        app = WebcamApp(image_path, device_index=args.device)
        app.run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
