"""Real ground data for each city: elevation and land cover on a 120 x 120 grid of 100 m cells (12 km x 12 km).

Sources (free, public):
* Elevation: AWS Terrain Tiles ("terrarium" encoding; SRTM and other open elevation models, about 30 m),
  averaged over each grid cell.
* Land cover: Sentinel-2 10 m Land Use / Land Cover (Esri, Impact Observatory, Microsoft; CC BY 4.0),
  read from Esri's public image service as a class-coloured image.

Grids are saved under data/terrain/, so the app and the tests run offline.

    python -m baadal.geodata          # download every city that is not saved yet
"""

from __future__ import annotations

import io
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .weather import CITIES, CITY_BY_ID

TERRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "terrain"
ELEVATION_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_ZOOM = 12
LANDCOVER_URL = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer/exportImage"
N = 120
HALF_KM = 6.0

# Class colours of the Sentinel-2 10 m land-cover rendering.
LC_COLOURS = {
    "water": (26, 91, 171), "trees": (53, 130, 33), "flooded_vegetation": (135, 209, 158), "crops": (255, 219, 92),
    "built": (237, 2, 42), "bare": (237, 233, 228), "snow": (242, 250, 255), "clouds": (200, 200, 200), "rangeland": (239, 207, 168),
}
LC_NAMES = list(LC_COLOURS)
LC = {name: k for k, name in enumerate(LC_NAMES)}


def bbox_for(city: dict) -> tuple[float, float, float, float]:
    """(south, west, north, east) of a square about 12 km wide centred on the city."""
    dlat = HALF_KM / 110.574
    dlon = HALF_KM / (111.320 * math.cos(math.radians(city["lat"])))
    return city["lat"] - dlat, city["lon"] - dlon, city["lat"] + dlat, city["lon"] + dlon


def fetch_site(city_id: str) -> dict:
    city = CITY_BY_ID[city_id]
    s, w, n, e = bbox_for(city)
    lats = [n - (r + 0.5) * (n - s) / N for r in range(N)]          # row 0 is the northern edge
    lons = [w + (c + 0.5) * (e - w) / N for c in range(N)]
    from PIL import Image
    elev = _tile_elevation(lats, lons, (n - s) / N, (e - w) / N)
    q = urllib.parse.urlencode({"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326, "size": f"{N * 3},{N * 3}",
                                "format": "png", "f": "image"}, safe=",")
    req = urllib.request.Request(f"{LANDCOVER_URL}?{q}", headers={"User-Agent": "baadal-flood-planner/0.3"})
    with urllib.request.urlopen(req, timeout=60) as r:
        img = np.array(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(int)
    cols = np.array([LC_COLOURS[name] for name in LC_NAMES])
    cls = np.argmin(((img[:, :, None, :] - cols[None, None, :, :]) ** 2).sum(axis=3), axis=2)      # nearest class colour
    # Each model cell covers 3 x 3 image pixels: store the fraction of each class and the majority class.
    blocks = cls.reshape(N, 3, N, 3).transpose(0, 2, 1, 3).reshape(N, N, 9)
    frac = np.stack([(blocks == k).mean(axis=2) for k in range(len(LC_NAMES))], axis=2)
    site = {
        "city_id": city_id, "bbox": [s, w, n, e], "n": N,
        "elevation_m": np.round(elev, 2).tolist(),
        "landcover": frac.argmax(axis=2).tolist(),
        "built_fraction": frac[:, :, LC["built"]].round(2).tolist(),
        "water_fraction": frac[:, :, LC["water"]].round(2).tolist(),
        "classes": LC_NAMES,
        "sources": "AWS Terrain Tiles (SRTM and other open elevation models); Sentinel-2 10 m Land Use/Land Cover (Esri, Impact Observatory, Microsoft)",
    }
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    (TERRAIN_DIR / f"{city_id}.json").write_text(json.dumps(site), encoding="utf-8")
    return site


_TILES: dict = {}


def _tile(z: int, x: int, y: int) -> np.ndarray:
    """Decoded elevation (m) of one 256 x 256 terrarium tile."""
    if (z, x, y) not in _TILES:
        from PIL import Image
        req = urllib.request.Request(ELEVATION_TILES.format(z=z, x=x, y=y), headers={"User-Agent": "baadal-flood-planner/0.3"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rgb = np.array(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(float)
        _TILES[(z, x, y)] = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
    return _TILES[(z, x, y)]


def _pixel(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Global web-mercator pixel coordinates (x, y) at zoom z."""
    scale = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * scale
    return x, y


def _tile_elevation(lats, lons, dlat: float, dlon: float) -> np.ndarray:
    """Mean elevation of each grid cell, sampled on a 3 x 3 pattern inside the cell."""
    out = np.zeros((len(lats), len(lons)))
    offs = np.linspace(-0.33, 0.33, 3)
    for r, la in enumerate(lats):
        for c, lo in enumerate(lons):
            vals = []
            for a in offs:
                for b in offs:
                    px, py = _pixel(la + a * dlat, lo + b * dlon, TILE_ZOOM)
                    t = _tile(TILE_ZOOM, int(px // 256), int(py // 256))
                    vals.append(t[int(py % 256), int(px % 256)])
            out[r, c] = float(np.mean(vals))
    return out


def has_site(city_id: str) -> bool:
    return (TERRAIN_DIR / f"{city_id}.json").exists()


def load_site(city_id: str) -> dict:
    return json.loads((TERRAIN_DIR / f"{city_id}.json").read_text(encoding="utf-8"))


def main():
    for c in CITIES:
        if has_site(c["id"]):
            continue
        site = fetch_site(c["id"])
        lc = np.array(site["landcover"])
        e = np.array(site["elevation_m"])
        print(f"{c['id']:11s} elevation {e.min():.0f}-{e.max():.0f} m | " +
              ", ".join(f"{name} {100 * (lc == k).mean():.0f}%" for k, name in enumerate(LC_NAMES) if (lc == k).mean() > 0.02), flush=True)


if __name__ == "__main__":
    main()
