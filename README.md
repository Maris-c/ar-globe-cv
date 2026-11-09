# AR Globe CV

A Python + OpenCV prototype for equirectangular map exploration and augmented reality research.

> **Milestone status:** M1 ✅ · M2 ✅ · M3 ✅ · M4 ✅ · M5 ✅ · M6 ✅

---

## Features

| Mode | Description |
|------|-------------|
| **Image Map** | Click any location on a static map image → get pixel coords + lat/lon |
| **Webcam Map** | Point webcam at a printed/displayed map → detect it with ORB + RANSAC → click to get lat/lon |

---

## Requirements

- Windows 10/11
- Python 3.11+
- OpenCV (`opencv-python`) ≥ 4.8
- NumPy ≥ 1.24
- pytest (for tests only)

---

## Installation

```powershell
# 1. Clone / open the project
cd "d:\NCKH\AR GLOBE\ar-globe-cv"

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest   # for running tests

# 4. Add your world map image
#    See assets/README.md for recommended sources.
#    Place the file at:  assets/images/world_map.jpg
```

---

## Running

### Interactive menu (recommended for first run)

```powershell
python src/main.py
```

### Image Map mode — direct

```powershell
python src/main.py --mode image --map assets/images/world_map.jpg
# or run the module directly:
python -m src.image_map.image_map_app --map assets/images/world_map.jpg
```

### Webcam Map mode — direct

```powershell
python src/main.py --mode webcam --map assets/images/world_map.jpg --device 0
```

If your webcam is not device 0, try `--device 1`, `--device 2`, etc.

---

## Keyboard Shortcuts

### Image Map window

| Key | Action |
|-----|--------|
| Left-click | Place marker, print lat/lon |
| Right-click | Remove last marker |
| `Z` | Undo last marker |
| `C` | Clear all markers |
| `S` | Save points to CSV in `outputs/` |
| `R` | Reset window size |
| `Q` / `ESC` | Quit |

### Webcam Map window

| Key | Action |
|-----|--------|
| Left-click | Place marker on detected map |
| `Z` | Undo last marker |
| `C` | Clear all markers |
| `S` | Save points to CSV in `outputs/` |
| `D` | Toggle debug overlay (match/inlier counts) |
| `Q` / `ESC` | Quit |

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

Expected output: **all tests pass** (38 coordinate tests + homography tests).

---

## Project Structure

```
ar-globe-cv/
├── assets/
│   ├── images/          ← Place world_map.jpg here
│   └── README.md        ← Image sourcing guide
├── configs/
│   └── default.json     ← All tuneable parameters
├── src/
│   ├── main.py          ← Top-level launcher
│   ├── config.py        ← Config loader
│   ├── coordinates/
│   │   └── map_coordinates.py   ← pixel↔lat/lon conversion
│   ├── image_map/
│   │   └── image_map_app.py     ← Mode 1
│   ├── webcam/
│   │   ├── webcam_app.py        ← Mode 2
│   │   ├── feature_matcher.py   ← ORB + BFMatcher
│   │   └── homography.py        ← Estimation + validation
│   ├── rendering/
│   │   └── overlay.py           ← Drawing helpers
│   └── utils/
│       └── file_utils.py        ← I/O utilities
├── tests/
│   ├── test_coordinates.py
│   └── test_homography.py
├── outputs/             ← CSV exports land here
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `configs/default.json` to tune behaviour without changing code:

```json
{
  "feature_matching": {
    "orb_n_features": 1500,    // more → slower but more robust
    "lowe_ratio": 0.75,        // lower → stricter match filter
    "min_matches": 15,         // raise to reduce false positives
    "min_inliers": 10,
    "min_inlier_ratio": 0.25
  }
}
```

---

## Coordinate System

> **⚠ Baseline Assumption**
>
> All lat/lon values assume your map is an **equirectangular (plate carrée)** projection
> covering the full world (−180° to +180° longitude, −90° to +90° latitude).
>
> For any other projection or regional map, coordinates will be **incorrect**.
> The `MapConfig` class is designed for future extension with custom extents and projections.

Conversion formulas:
- `pixel_to_latlon(x, y, width, height)` → `(latitude, longitude)`
- `latlon_to_pixel(lat, lon, width, height)` → `(x, y)`

---

## Known Limitations

1. **Projection**: Only equirectangular is implemented. Mercator, Robinson, etc. not supported.
2. **Homography = planar**: ORB + RANSAC works well for flat printed maps; it does not handle a real 3D globe.
3. **Lighting**: Detection degrades under low light, glare, or partial occlusion.
4. **Screen detection**: Maps displayed on a glossy screen may trigger fewer ORB features due to reflections; matte paper works better.
5. **OpenCV 5.x**: Project was built with OpenCV 5.0.0.93; API changes may affect older versions.
6. **⚠ Webcam benchmark disclaimer**: Detection rates, FPS, and tracking statistics have **not been independently verified on external hardware**. Numbers shown during a session are from that specific run only.

---

## Roadmap (post-MVP)

| Phase | Goal |
|-------|------|
| v0.2 | PySide6 GUI with proper buttons and panels |
| v0.3 | SIFT/SuperPoint as alternative feature detectors |
| v0.4 | Camera calibration + undistortion |
| v0.5 | Cylindrical map projection support |
| v1.0 | Physical globe tracking with sphere-aware pose estimation |

---

## Demo Checklist

- [ ] `assets/images/world_map.jpg` present
- [ ] `python -m pytest tests/ -v` → all green
- [ ] `python src/main.py --mode image --map assets/images/world_map.jpg` opens window
- [ ] Click on map → lat/lon printed in terminal
- [ ] Press `S` → CSV saved to `outputs/`
- [ ] `python src/main.py --mode webcam --map assets/images/world_map.jpg` opens webcam
- [ ] Hold printed map in front of camera → green polygon appears
- [ ] Click inside polygon → lat/lon printed

---

*Built as a research prototype. Not production software.*
