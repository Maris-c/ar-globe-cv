"""
image_map_app.py — Mode 1: Interactive image map with click-to-coordinate.

Usage (standalone):
    python -m src.image_map.image_map_app --map assets/images/world_map.jpg

Keyboard shortcuts inside the OpenCV window:
    Left-click  → Place marker and print coordinates
    Right-click → Remove the last placed marker
    S           → Save all points to CSV
    C           → Clear all markers
    Z           → Undo last marker
    R           → Reset view (zoom/pan)
    Q / ESC     → Quit
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Allow running as standalone script from project root.
_SRC = pathlib.Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CFG
from coordinates.map_coordinates import MapConfig, pixel_to_latlon
from rendering.overlay import (
    draw_marker, draw_coord_label, draw_info_panel, draw_status_bar,
    DARK_BG, GREEN, RED, YELLOW, WHITE, CYAN,
)
from utils.file_utils import load_image, save_points_csv, choose_image_file_dialog


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SelectedPoint:
    """A single user-selected point on the map."""
    pixel_x: int          # in original image coordinates
    pixel_y: int
    latitude: float
    longitude: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class ImageMapApp:
    """Interactive equirectangular map viewer with click-to-latlon support.

    Parameters
    ----------
    image_path:
        Path to the map image file.
    map_cfg:
        Geographic extent / projection configuration.
    window_name:
        OpenCV window title.
    max_display_width:
        If the image is wider than this, it is scaled down for display.
        The original image coordinates are always used for coordinate math.
    """

    WINDOW = "AR Globe CV — Image Map"
    SHORTCUTS = [
        "LClick  → Place marker",
        "RClick  → Remove last",
        "Z       → Undo",
        "C       → Clear all",
        "S       → Save CSV",
        "R       → Reset view",
        "Q/ESC   → Quit",
    ]

    def __init__(
        self,
        image_path: str | pathlib.Path,
        map_cfg: Optional[MapConfig] = None,
        max_display_width: int = 1280,
    ) -> None:
        self._img_orig: np.ndarray = load_image(image_path)
        self._h, self._w = self._img_orig.shape[:2]
        self._map_cfg = map_cfg or MapConfig(
            projection=CFG["map"]["projection"],
            lon_min=CFG["map"]["lon_min"],
            lon_max=CFG["map"]["lon_max"],
            lat_min=CFG["map"]["lat_min"],
            lat_max=CFG["map"]["lat_max"],
        )

        # Compute display scale so large images fit on screen.
        self._scale: float = min(1.0, max_display_width / self._w)
        self._disp_w = int(self._w * self._scale)
        self._disp_h = int(self._h * self._scale)

        self._points: list[SelectedPoint] = []
        self._hover_pos: Optional[tuple[int, int]] = None   # display coords
        self._status_msg: str = "Ready  —  Left-click to select a location"

        print(f"[ImageMapApp] Loaded {image_path}")
        print(f"  Image size : {self._w} x {self._h} px")
        print(f"  Display    : {self._disp_w} x {self._disp_h} px  (scale={self._scale:.3f})")
        print(f"  Projection : {self._map_cfg.projection}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the window and enter the event loop."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self._disp_w, self._disp_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse_cb)

        print(f"\n[ImageMapApp] Window opened. Shortcuts:")
        for s in self.SHORTCUTS:
            print(f"  {s}")
        print()

        while True:
            frame = self._render_frame()
            cv2.imshow(self.WINDOW, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("s"), ord("S")):
                self._save_csv()
            elif key in (ord("c"), ord("C")):
                self._clear_all()
            elif key in (ord("z"), ord("Z")):
                self._undo()
            elif key in (ord("r"), ord("R")):
                cv2.resizeWindow(self.WINDOW, self._disp_w, self._disp_h)

            # Check if window was closed by the user via the X button.
            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyWindow(self.WINDOW)
        print("[ImageMapApp] Window closed.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        """Build the annotated display frame."""
        # Always resize original to current display size so window resizes work.
        win_rect = cv2.getWindowImageRect(self.WINDOW)
        if win_rect[2] > 0 and win_rect[3] > 0:
            dw, dh = win_rect[2], win_rect[3]
        else:
            dw, dh = self._disp_w, self._disp_h

        frame = cv2.resize(self._img_orig, (dw, dh), interpolation=cv2.INTER_LINEAR)
        scale_x = dw / self._w
        scale_y = dh / self._h

        # Draw all placed markers.
        for i, pt in enumerate(self._points):
            dx = int(pt.pixel_x * scale_x)
            dy = int(pt.pixel_y * scale_y)
            draw_marker(frame, dx, dy, index=i + 1)
            draw_coord_label(frame, dx, dy, pt.latitude, pt.longitude, pt.pixel_x, pt.pixel_y)

        # Info panel (left side).
        draw_info_panel(frame, [p.to_dict() for p in self._points],
                        status="", shortcuts=self.SHORTCUTS)

        # Status bar.
        draw_status_bar(frame, self._status_msg)

        # Image metadata top-right.
        meta = f"{self._w}x{self._h}px | {self._map_cfg.projection}"
        cv2.putText(frame, meta, (dw - 320, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        win_rect = cv2.getWindowImageRect(self.WINDOW)
        dw = win_rect[2] if win_rect[2] > 0 else self._disp_w
        dh = win_rect[3] if win_rect[3] > 0 else self._disp_h

        # Map display pixel → original image pixel.
        orig_x = int(x * self._w / dw)
        orig_y = int(y * self._h / dh)
        orig_x = max(0, min(self._w - 1, orig_x))
        orig_y = max(0, min(self._h - 1, orig_y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._place_point(orig_x, orig_y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._undo()
        elif event == cv2.EVENT_MOUSEMOVE:
            self._hover_pos = (x, y)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _place_point(self, orig_x: int, orig_y: int) -> None:
        lat, lon = pixel_to_latlon(orig_x, orig_y, self._w, self._h, self._map_cfg)
        pt = SelectedPoint(pixel_x=orig_x, pixel_y=orig_y, latitude=lat, longitude=lon)
        self._points.append(pt)

        msg = (f"  #{len(self._points):>3} | pixel=({orig_x:5d},{orig_y:5d}) | "
               f"lat={lat:+8.4f}°  lon={lon:+9.4f}°")
        print(msg)
        self._status_msg = f"Point #{len(self._points)}: lat={lat:+.4f}°  lon={lon:+.4f}°"

    def _undo(self) -> None:
        if self._points:
            removed = self._points.pop()
            self._status_msg = f"Removed point #{len(self._points)+1}"
            print(f"[ImageMapApp] Removed: pixel=({removed.pixel_x},{removed.pixel_y})")
        else:
            self._status_msg = "No points to remove."

    def _clear_all(self) -> None:
        n = len(self._points)
        self._points.clear()
        self._status_msg = f"Cleared {n} point(s)."
        print(f"[ImageMapApp] Cleared {n} point(s).")

    def _save_csv(self) -> None:
        if not self._points:
            self._status_msg = "No points to save."
            return
        try:
            out_dir = pathlib.Path(CFG["output"]["csv_dir"])
            path = save_points_csv([p.to_dict() for p in self._points], out_dir)
            self._status_msg = f"Saved {len(self._points)} points → {path.name}"
            print(f"[ImageMapApp] CSV saved: {path.resolve()}")
        except Exception as exc:  # noqa: BLE001
            self._status_msg = f"Save failed: {exc}"
            print(f"[ImageMapApp] ERROR saving CSV: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AR Globe CV — Image Map Mode",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--map", "-m",
        type=str,
        default=None,
        help="Path to the world map image (PNG/JPG). Opens file dialog if omitted.",
    )
    args = parser.parse_args()

    image_path = args.map
    if image_path is None:
        print("[ImageMapApp] No --map argument provided. Opening file dialog...")
        image_path = choose_image_file_dialog()

    if not image_path:
        print("[ImageMapApp] No image selected. Exiting.")
        sys.exit(1)

    try:
        app = ImageMapApp(image_path)
        app.run()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
