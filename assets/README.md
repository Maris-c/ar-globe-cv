# assets/images/

Place your world map image here before running the app.

## Recommended file

| Property | Value |
|----------|-------|
| Filename | `world_map.jpg` or `world_map.png` |
| Projection | **Equirectangular** (plate carrée) |
| Coverage | Full world: −180° to +180° lon, −90° to +90° lat |
| Resolution | ≥ 2048 × 1024 px recommended |

## Where to get a free equirectangular world map

- **Natural Earth** (public domain): https://www.naturalearthdata.com/downloads/
- **NASA Blue Marble**: https://visibleearth.nasa.gov/collection/1484/blue-marble
- **OpenStreetMap** raster exports via tools like TileMill or MapTiler

## Checking your image

Run from the project root:

```
python -c "import cv2; img=cv2.imread('assets/images/world_map.jpg'); print(img.shape)"
```

Expected output: `(height, width, 3)` — e.g. `(1024, 2048, 3)` for a 2:1 ratio.

## Important

The coordinate math assumes a **true equirectangular projection**.
If your map uses a different projection (Mercator, Robinson, etc.)
the lat/lon values will be incorrect. The code labels this explicitly.
