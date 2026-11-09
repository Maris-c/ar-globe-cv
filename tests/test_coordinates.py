"""
test_coordinates.py — Unit tests for src/coordinates/map_coordinates.py

Run with:
    python -m pytest tests/test_coordinates.py -v
or from the project root:
    python -m pytest -v
"""

import math
import sys
import pathlib
import pytest

# Allow running from project root without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from coordinates.map_coordinates import (
    MapConfig,
    DEFAULT_MAP_CONFIG,
    pixel_to_latlon,
    latlon_to_pixel,
    round_trip_error,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

W, H = 3600, 1800  # A common equirectangular map size


# ===========================================================================
# 1.  MapConfig validation
# ===========================================================================

class TestMapConfig:
    def test_default_values(self):
        cfg = MapConfig()
        assert cfg.lon_min == -180.0
        assert cfg.lon_max == 180.0
        assert cfg.lat_min == -90.0
        assert cfg.lat_max == 90.0
        assert cfg.lon_span == 360.0
        assert cfg.lat_span == 180.0

    def test_custom_extent(self):
        cfg = MapConfig(lon_min=0, lon_max=90, lat_min=10, lat_max=60)
        assert cfg.lon_span == 90.0
        assert cfg.lat_span == 50.0

    def test_invalid_lon_order_raises(self):
        with pytest.raises(ValueError, match="lon_min"):
            MapConfig(lon_min=90, lon_max=-90)

    def test_invalid_lat_order_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            MapConfig(lat_min=90, lat_max=-90)


# ===========================================================================
# 2.  pixel_to_latlon — corner tests
# ===========================================================================

class TestPixelToLatlon:
    """All corner / edge / centre cases for the default world map."""

    def _p2l(self, x, y):
        return pixel_to_latlon(x, y, W, H)

    # --- corners ---

    def test_top_left_corner(self):
        lat, lon = self._p2l(0, 0)
        assert lat == pytest.approx(90.0, abs=1e-4)
        assert lon == pytest.approx(-180.0, abs=1e-4)

    def test_top_right_corner(self):
        lat, lon = self._p2l(W, 0)
        assert lat == pytest.approx(90.0, abs=1e-4)
        assert lon == pytest.approx(180.0, abs=1e-4)

    def test_bottom_left_corner(self):
        lat, lon = self._p2l(0, H)
        assert lat == pytest.approx(-90.0, abs=1e-4)
        assert lon == pytest.approx(-180.0, abs=1e-4)

    def test_bottom_right_corner(self):
        lat, lon = self._p2l(W, H)
        assert lat == pytest.approx(-90.0, abs=1e-4)
        assert lon == pytest.approx(180.0, abs=1e-4)

    # --- centre ---

    def test_centre_pixel(self):
        lat, lon = self._p2l(W / 2, H / 2)
        assert lat == pytest.approx(0.0, abs=0.1)
        assert lon == pytest.approx(0.0, abs=0.1)

    # --- edge midpoints ---

    def test_top_edge_midpoint(self):
        lat, lon = self._p2l(W / 2, 0)
        assert lat == pytest.approx(90.0, abs=1e-4)
        assert lon == pytest.approx(0.0, abs=0.1)

    def test_bottom_edge_midpoint(self):
        lat, lon = self._p2l(W / 2, H)
        assert lat == pytest.approx(-90.0, abs=1e-4)
        assert lon == pytest.approx(0.0, abs=0.1)

    def test_left_edge_midpoint(self):
        lat, lon = self._p2l(0, H / 2)
        assert lat == pytest.approx(0.0, abs=0.1)
        assert lon == pytest.approx(-180.0, abs=1e-4)

    def test_right_edge_midpoint(self):
        lat, lon = self._p2l(W, H / 2)
        assert lat == pytest.approx(0.0, abs=0.1)
        assert lon == pytest.approx(180.0, abs=1e-4)

    # --- input validation ---

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x="):
            pixel_to_latlon(-1, 0, W, H)

    def test_x_beyond_width_raises(self):
        with pytest.raises(ValueError, match="x="):
            pixel_to_latlon(W + 1, 0, W, H)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y="):
            pixel_to_latlon(0, -1, W, H)

    def test_y_beyond_height_raises(self):
        with pytest.raises(ValueError, match="y="):
            pixel_to_latlon(0, H + 1, W, H)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            pixel_to_latlon(0, 0, 0, H)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            pixel_to_latlon(0, 0, W, 0)

    # --- unsupported projection ---

    def test_unsupported_projection_raises(self):
        cfg = MapConfig(projection="mercator")
        with pytest.raises(NotImplementedError, match="mercator"):
            pixel_to_latlon(0, 0, W, H, cfg)


# ===========================================================================
# 3.  latlon_to_pixel — corner and known-coordinate tests
# ===========================================================================

class TestLatlonToPixel:
    def _l2p(self, lat, lon):
        return latlon_to_pixel(lat, lon, W, H)

    def test_north_pole_left(self):
        x, y = self._l2p(90.0, -180.0)
        assert x == 0
        assert y == 0

    def test_north_pole_right(self):
        x, y = self._l2p(90.0, 180.0)
        assert x == W - 1  # clamped to width-1

    def test_south_pole_left(self):
        x, y = self._l2p(-90.0, -180.0)
        assert x == 0
        assert y == H - 1

    def test_equator_prime_meridian(self):
        x, y = self._l2p(0.0, 0.0)
        assert x == pytest.approx(W // 2, abs=2)
        assert y == pytest.approx(H // 2, abs=2)

    def test_invalid_latitude_raises(self):
        with pytest.raises(ValueError, match="latitude"):
            latlon_to_pixel(91.0, 0.0, W, H)

    def test_invalid_longitude_raises(self):
        with pytest.raises(ValueError, match="longitude"):
            latlon_to_pixel(0.0, 181.0, W, H)

    def test_result_always_in_bounds(self):
        """pixel result must never exceed image dimensions."""
        for lat, lon in [(90, 180), (-90, -180), (0, 0), (45, 90), (-45, -90)]:
            x, y = self._l2p(lat, lon)
            assert 0 <= x < W, f"x={x} out of range for lat={lat}, lon={lon}"
            assert 0 <= y < H, f"y={y} out of range for lat={lat}, lon={lon}"


# ===========================================================================
# 4.  Round-trip fidelity tests (pixel → latlon → pixel)
# ===========================================================================

class TestRoundTrip:
    """Verify that converting pixel→latlon→pixel stays within ~1 pixel."""

    SAMPLE_PIXELS = [
        (0, 0),
        (W, 0),
        (0, H),
        (W, H),
        (W // 2, H // 2),
        (100, 200),
        (3000, 1500),
        (1800, 900),
    ]

    @pytest.mark.parametrize("x,y", SAMPLE_PIXELS)
    def test_round_trip_sub_pixel(self, x, y):
        err = round_trip_error(x, y, W, H)
        assert err < 1.5, (
            f"Round-trip error {err:.3f}px at pixel ({x},{y}) exceeds 1.5px tolerance."
        )

    def test_round_trip_latlon_to_pixel(self):
        """latlon → pixel → latlon should yield < 0.05° error."""
        for lat, lon in [(45.0, 90.0), (-23.5, -46.6), (51.5, -0.1), (35.7, 139.7)]:
            px, py = latlon_to_pixel(lat, lon, W, H)
            lat2, lon2 = pixel_to_latlon(px, py, W, H)
            assert abs(lat2 - lat) < 0.05, f"Lat error too large: {lat2} vs {lat}"
            assert abs(lon2 - lon) < 0.05, f"Lon error too large: {lon2} vs {lon}"


# ===========================================================================
# 5.  Custom MapConfig extent
# ===========================================================================

class TestCustomExtent:
    """Verify that non-default extents work correctly."""

    CFG = MapConfig(lon_min=0.0, lon_max=90.0, lat_min=0.0, lat_max=45.0)
    W2, H2 = 900, 450

    def test_top_left_origin(self):
        lat, lon = pixel_to_latlon(0, 0, self.W2, self.H2, self.CFG)
        assert lat == pytest.approx(45.0, abs=1e-4)
        assert lon == pytest.approx(0.0, abs=1e-4)

    def test_bottom_right_extent(self):
        lat, lon = pixel_to_latlon(self.W2, self.H2, self.W2, self.H2, self.CFG)
        assert lat == pytest.approx(0.0, abs=1e-4)
        assert lon == pytest.approx(90.0, abs=1e-4)
