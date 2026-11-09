"""
map_coordinates.py — Pixel ↔ Latitude/Longitude conversion for equirectangular maps.

IMPORTANT DISCLAIMER
--------------------
The functions in this module implement a **baseline equirectangular mapping**:

    top-left  pixel  →  (lon=-180°, lat=+90°)
    top-right pixel  →  (lon=+180°, lat=+90°)
    bot-left  pixel  →  (lon=-180°, lat=-90°)
    bot-right pixel  →  (lon=+180°, lat=-90°)

This is a valid approximation ONLY when:
  1. The map image is a true equirectangular (plate carrée) projection.
  2. The map covers the full world extent (−180° to +180° lon, −90° to +90° lat).

Do NOT assert geographic accuracy for:
  - Mercator, Robinson, Winkel-Tripel, or any other projection.
  - Cropped or regionally-bounded maps.

The ``MapConfig`` dataclass allows the caller to supply custom extents and a
projection label so the limitation is always explicit.  Additional projection
transforms can be added in future milestones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MapConfig:
    """Describes the geographic extent and projection of a map image.

    Attributes
    ----------
    projection:
        Human-readable projection name.  Only ``"equirectangular"`` is
        currently implemented; any other value raises ``NotImplementedError``.
    lon_min, lon_max:
        Longitude extent in decimal degrees.
    lat_min, lat_max:
        Latitude extent in decimal degrees.
    """

    projection: str = "equirectangular"
    lon_min: float = -180.0
    lon_max: float = 180.0
    lat_min: float = -90.0
    lat_max: float = 90.0

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})")

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min


# Default world config — shared across the project.
DEFAULT_MAP_CONFIG = MapConfig()


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def pixel_to_latlon(
    x: float,
    y: float,
    width: int,
    height: int,
    cfg: MapConfig = DEFAULT_MAP_CONFIG,
) -> tuple[float, float]:
    """Convert pixel coordinates to (latitude, longitude).

    Parameters
    ----------
    x, y:
        Pixel coordinates.  The origin (0, 0) is the **top-left** corner.
        ``x`` increases rightward; ``y`` increases downward.
    width, height:
        Image dimensions in pixels (must be > 0).
    cfg:
        Geographic extent configuration.

    Returns
    -------
    (latitude, longitude) in decimal degrees.

    Raises
    ------
    ValueError
        If *width* or *height* ≤ 0, or if *x*/*y* are outside [0, width/height].
    NotImplementedError
        If ``cfg.projection`` is not ``"equirectangular"``.
    """
    _validate_dimensions(width, height)
    _validate_pixel(x, y, width, height)
    _require_equirectangular(cfg)

    # Normalise to [0, 1] — clamp to handle exact-boundary clicks robustly.
    nx = max(0.0, min(1.0, x / width))
    ny = max(0.0, min(1.0, y / height))

    longitude = cfg.lon_min + nx * cfg.lon_span
    # y=0 → lat_max (north pole);  y=height → lat_min (south pole)
    latitude = cfg.lat_max - ny * cfg.lat_span

    return round(latitude, 6), round(longitude, 6)


def latlon_to_pixel(
    latitude: float,
    longitude: float,
    width: int,
    height: int,
    cfg: MapConfig = DEFAULT_MAP_CONFIG,
) -> tuple[int, int]:
    """Convert (latitude, longitude) to pixel coordinates.

    Parameters
    ----------
    latitude:
        Decimal degrees, clamped to [lat_min, lat_max].
    longitude:
        Decimal degrees, clamped to [lon_min, lon_max].
    width, height:
        Image dimensions in pixels (must be > 0).
    cfg:
        Geographic extent configuration.

    Returns
    -------
    (x, y) pixel coordinates (integers, origin top-left).

    Raises
    ------
    ValueError
        If *width* or *height* ≤ 0, or if lat/lon are out of valid global range.
    NotImplementedError
        If ``cfg.projection`` is not ``"equirectangular"``.
    """
    _validate_dimensions(width, height)
    _validate_latlon(latitude, longitude)
    _require_equirectangular(cfg)

    # Clamp to configured extents (graceful handling of boundary values).
    lat = max(cfg.lat_min, min(cfg.lat_max, latitude))
    lon = max(cfg.lon_min, min(cfg.lon_max, longitude))

    nx = (lon - cfg.lon_min) / cfg.lon_span
    ny = (cfg.lat_max - lat) / cfg.lat_span  # y increases downward

    x = int(round(nx * width))
    y = int(round(ny * height))

    # Clamp to [0, dim-1] so we never produce out-of-bounds pixels.
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


# ---------------------------------------------------------------------------
# Inverse-trip helper (round-trip fidelity check)
# ---------------------------------------------------------------------------

def round_trip_error(
    x: float,
    y: float,
    width: int,
    height: int,
    cfg: MapConfig = DEFAULT_MAP_CONFIG,
) -> float:
    """Return the Euclidean pixel distance after a pixel→latlon→pixel round trip.

    A value < 1.0 means sub-pixel accuracy, which is the expected result for
    equirectangular maps with integer pixel dimensions.
    """
    lat, lon = pixel_to_latlon(x, y, width, height, cfg)
    rx, ry = latlon_to_pixel(lat, lon, width, height, cfg)
    return math.hypot(rx - x, ry - y)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive; got width={width}, height={height}.")


def _validate_pixel(x: float, y: float, width: int, height: int) -> None:
    if not (0 <= x <= width):
        raise ValueError(f"x={x} is outside [0, {width}].")
    if not (0 <= y <= height):
        raise ValueError(f"y={y} is outside [0, {height}].")


def _validate_latlon(latitude: float, longitude: float) -> None:
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"latitude={latitude} is outside [-90, 90].")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"longitude={longitude} is outside [-180, 180].")


def _require_equirectangular(cfg: MapConfig) -> None:
    if cfg.projection.lower() != "equirectangular":
        raise NotImplementedError(
            f"Projection '{cfg.projection}' is not yet implemented. "
            "Only 'equirectangular' is supported in this milestone."
        )
