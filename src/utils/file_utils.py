"""
file_utils.py — File I/O helpers: image loading, CSV export, path validation.
"""

from __future__ import annotations

import csv
import pathlib
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def load_image(path: str | pathlib.Path) -> np.ndarray:
    """Load an image from *path* using OpenCV.

    Parameters
    ----------
    path:
        Absolute or relative path to the image file.

    Returns
    -------
    BGR image as a NumPy array (uint8).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is not in SUPPORTED_IMAGE_EXTENSIONS or OpenCV
        fails to decode the file.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p.resolve()}")
    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(
            f"OpenCV could not decode the image: {p.resolve()}\n"
            "The file may be corrupted or in an unsupported sub-format."
        )
    return img


def save_points_csv(
    points: list[dict],
    output_dir: str | pathlib.Path = "outputs",
    filename: Optional[str] = None,
) -> pathlib.Path:
    """Save a list of selected map points to a CSV file.

    Each dict in *points* should contain keys: ``pixel_x``, ``pixel_y``,
    ``latitude``, ``longitude``.  Extra keys are written as additional columns.

    Parameters
    ----------
    points:
        List of point dicts to export.
    output_dir:
        Directory where the CSV will be saved (created if needed).
    filename:
        Override the auto-generated filename.

    Returns
    -------
    Path to the written CSV file.
    """
    if not points:
        raise ValueError("No points to save.")

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"points_{ts}.csv"

    csv_path = out / filename

    # Collect all field names (preserve insertion order, put standard ones first)
    standard_keys = ["pixel_x", "pixel_y", "latitude", "longitude"]
    extra_keys = [k for k in points[0].keys() if k not in standard_keys]
    fieldnames = standard_keys + extra_keys

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)

    return csv_path


def choose_image_file_dialog() -> Optional[str]:
    """Open a simple Tkinter file-chooser dialog and return the selected path.

    Returns ``None`` if the user cancelled.  Tkinter is used here to avoid
    adding a heavy GUI dependency; the dialog is destroyed immediately after use.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # hide the root window
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select Map Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as exc:  # noqa: BLE001
        print(f"[file_utils] File dialog unavailable: {exc}")
        return None
