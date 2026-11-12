"""Weather and river data collection (Open-Meteo, free, no API key).

Live mode   : 7-day rain forecast, a 31-member rain ensemble (GEFS) and GloFAS river-discharge
              forecasts with a 50-member ensemble, for a watch-list of Pakistani cities.
Replay mode : the same quantities for a real past event, taken from reanalysis archives
              (ERA5 / ECMWF IFS rain, GloFAS discharge). Replay is a hindcast: the "forecast"
              for the 7 days after the as-of date is what was later analysed, not what a
              forecaster saw that day.

Every snapshot is saved under data/, so the pipeline also runs offline and results are reproducible.

    python -m baadal.weather --build    # river reference statistics + all replay events
    python -m baadal.weather --live     # fetch and save today's live snapshot
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPLAY_DIR = DATA / "replay"
REFERENCE_FILE = DATA / "reference_discharge.json"
LIVE_FILE = DATA / "live_latest.json"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TZ = "Asia/Karachi"
WINDOW_DAYS = 7
REF_YEARS = (2000, 2023)

# Watch-list. "riverine" cities are threatened mainly by river discharge, "urban" ones by rainfall.
# Coordinates are approximate gauge / city-centre locations; river points are snapped to the
# nearest GloFAS main-channel cell when the reference file is built.
CITIES = [
    {"id": "nowshera", "name": "Nowshera", "river": "Kabul River", "province": "KP", "kind": "riverine", "lat": 34.01, "lon": 71.98},
    {"id": "marala", "name": "Sialkot (Marala)", "river": "Chenab", "province": "Punjab", "kind": "riverine", "lat": 32.67, "lon": 74.46},
    {"id": "trimmu", "name": "Jhang (Trimmu)", "river": "Chenab", "province": "Punjab", "kind": "riverine", "lat": 31.14, "lon": 72.15},
    {"id": "multan", "name": "Multan", "river": "Chenab", "province": "Punjab", "kind": "riverine", "lat": 30.20, "lon": 71.33},
    {"id": "taunsa", "name": "D.G. Khan (Taunsa)", "river": "Indus", "province": "Punjab", "kind": "riverine", "lat": 30.51, "lon": 70.85},
    {"id": "guddu", "name": "Kashmore (Guddu)", "river": "Indus", "province": "Sindh", "kind": "riverine", "lat": 28.42, "lon": 69.72},
    {"id": "sukkur", "name": "Sukkur", "river": "Indus", "province": "Sindh", "kind": "riverine", "lat": 27.70, "lon": 68.86},
    {"id": "kotri", "name": "Hyderabad (Kotri)", "river": "Indus", "province": "Sindh", "kind": "riverine", "lat": 25.44, "lon": 68.31},
    {"id": "rawalpindi", "name": "Rawalpindi", "river": "Nullah Lai", "province": "Punjab", "kind": "urban", "lat": 33.60, "lon": 73.05},
    {"id": "lahore", "name": "Lahore", "river": "city drains", "province": "Punjab", "kind": "urban", "lat": 31.55, "lon": 74.34},
    {"id": "larkana", "name": "Larkana", "river": "city drains", "province": "Sindh", "kind": "urban", "lat": 27.56, "lon": 68.21},
    {"id": "karachi", "name": "Karachi", "river": "storm nullahs", "province": "Sindh", "kind": "urban", "lat": 24.86, "lon": 67.01},
]
CITY_BY_ID = {c["id"]: c for c in CITIES}

# Real past events available for replay. as_of is the day the 7-day window starts.
EVENTS = [
    {"id": "2010-07-28_kp_superflood", "name": "2010 super flood - week of 28 July", "as_of": "2010-07-28"},
    {"id": "2010-08-06_indus_superflood", "name": "2010 super flood - week of 6 August", "as_of": "2010-08-06"},
    {"id": "2014-09-04_chenab", "name": "September 2014 floods - Punjab and Rawalpindi", "as_of": "2014-09-04"},
    {"id": "2022-07-23_karachi", "name": "July 2022 monsoon cloudburst - Karachi", "as_of": "2022-07-23"},
    {"id": "2022-08-17_sindh_rains", "name": "August 2022 monsoon rains - upper Sindh", "as_of": "2022-08-17"},
    {"id": "2025-08-25_punjab", "name": "August 2025 floods - Punjab", "as_of": "2025-08-25"},
]
EVENT_BY_ID = {e["id"]: e for e in EVENTS}


# --------------------------------------------------------------------------- HTTP
def _get(base: str, **params):
    url = base + "?" + urllib.parse.urlencode(params, safe=",")
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "baadal-flood-planner/0.2"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as ex:
            last = ex
            if ex.code == 429:            # free-tier rate limit (per minute): wait for the window to reset
                print("  rate limited by Open-Meteo, waiting 65 s...", flush=True)
                time.sleep(65)
            else:
                time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as ex:
            last = ex
            time.sleep(1.5 * (attempt + 1))
    raise ConnectionError(f"Open-Meteo request failed: {last}")


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _roll3_max(daily) -> float:
    d = np.array([0.0 if v is None else float(v) for v in daily])
    if len(d) < 3:
        return float(d.sum())
    return float(max(d[i:i + 3].sum() for i in range(len(d) - 2)))


def _clean(xs):
    return [None if v is None else round(float(v), 2) for v in xs]


# --------------------------------------------------------------------------- reference statistics
def build_reference() -> dict:
    """Snap each river city to the GloFAS main channel and store 2000-2023 annual maximum discharge."""
    out = {"source": "GloFAS v4 reanalysis via Open-Meteo Flood API", "years": list(REF_YEARS), "cities": {}}
    if REFERENCE_FILE.exists():           # resumable: keep cities already downloaded
        out = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    DATA.mkdir(exist_ok=True)
    for c in CITIES:
        if c["kind"] != "riverine" or c["id"] in out["cities"]:
            continue
        pts = [(round(c["lat"] + i * 0.05, 3), round(c["lon"] + j * 0.05, 3)) for i in range(-2, 3) for j in range(-2, 3)]
        grid = _as_list(_get(FLOOD_URL, latitude=",".join(str(p[0]) for p in pts), longitude=",".join(str(p[1]) for p in pts),
                             start_date="2010-07-01", end_date="2010-09-30", daily="river_discharge"))
        best = max(grid, key=lambda g: max((v for v in g["daily"]["river_discharge"] if v is not None), default=0.0))
        lat, lon = best["latitude"], best["longitude"]
        hist = _get(FLOOD_URL, latitude=lat, longitude=lon, start_date=f"{REF_YEARS[0]}-01-01",
                    end_date=f"{REF_YEARS[1]}-12-31", daily="river_discharge")
        maxima: dict[str, float] = {}
        for t, q in zip(hist["daily"]["time"], hist["daily"]["river_discharge"]):
            if q is not None:
                maxima[t[:4]] = max(maxima.get(t[:4], 0.0), float(q))
        am = np.array(sorted(maxima.values()))
        out["cities"][c["id"]] = {
            "lat": lat, "lon": lon,
            "annual_max_m3s": {k: round(v, 1) for k, v in sorted(maxima.items())},
            "q2_m3s": round(float(np.quantile(am, 0.5)), 1),     # ~2-year flood (median annual maximum)
            "q5_m3s": round(float(np.quantile(am, 0.8)), 1),     # ~5-year flood
            "q10_m3s": round(float(np.quantile(am, 0.9)), 1),    # ~10-year flood
            "qmax_m3s": round(float(am.max()), 1),
        }
        print(f"  reference {c['id']:10s} cell ({lat:.3f},{lon:.3f}) Q2={out['cities'][c['id']]['q2_m3s']:.0f} "
              f"Q10={out['cities'][c['id']]['q10_m3s']:.0f} max={am.max():.0f} m3/s", flush=True)
        REFERENCE_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_reference() -> dict:
    return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))


def _river_points(ref: dict):
    ids = [c["id"] for c in CITIES if c["kind"] == "riverine"]
    return ids, ",".join(str(ref["cities"][i]["lat"]) for i in ids), ",".join(str(ref["cities"][i]["lon"]) for i in ids)


# --------------------------------------------------------------------------- live
def fetch_live(save: bool = True) -> dict:
    ref = load_reference()
    lats = ",".join(str(c["lat"]) for c in CITIES)
    lons = ",".join(str(c["lon"]) for c in CITIES)
    fc = _as_list(_get(FORECAST_URL, latitude=lats, longitude=lons, daily="precipitation_sum",
                       forecast_days=WINDOW_DAYS, timezone=TZ))
    ens = _as_list(_get(ENSEMBLE_URL, latitude=lats, longitude=lons, hourly="precipitation", models="gfs_seamless",
                        forecast_days=WINDOW_DAYS, timezone=TZ))
    rids, rlat, rlon = _river_points(ref)
    fl = _as_list(_get(FLOOD_URL, latitude=rlat, longitude=rlon, daily="river_discharge", forecast_days=WINDOW_DAYS, ensemble="true"))
    fl_by = dict(zip(rids, fl))

    snap = {"mode": "live", "as_of": fc[0]["daily"]["time"][0], "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "event": None, "source": "Open-Meteo forecast API (best-match model), GEFS 31-member ensemble, GloFAS flood forecast (50 members)",
            "cities": {}}
    for c, f, e in zip(CITIES, fc, ens):
        members = []
        for key, series in e["hourly"].items():
            if key.startswith("precipitation"):
                hourly = np.array([0.0 if v is None else float(v) for v in series])
                daily = hourly[: len(hourly) // 24 * 24].reshape(-1, 24).sum(axis=1)
                members.append(round(_roll3_max(daily), 2))
        entry = {"days": f["daily"]["time"], "rain_daily_mm": _clean(f["daily"]["precipitation_sum"]),
                 "rain_members_72h_mm": members, "rain_member_source": f"GEFS ensemble ({len(members)} members)",
                 "discharge_daily_m3s": None, "discharge_members_peak_m3s": None, "discharge_member_source": None}
        if c["id"] in fl_by:
            d = fl_by[c["id"]]["daily"]
            peaks = [round(max(v for v in series if v is not None), 1) for key, series in d.items()
                     if key.startswith("river_discharge_member") and any(v is not None for v in series)]
            entry.update({"discharge_daily_m3s": _clean(d["river_discharge"]), "discharge_members_peak_m3s": peaks,
                          "discharge_member_source": f"GloFAS ensemble ({len(peaks)} members)"})
        snap["cities"][c["id"]] = entry
    if save:
        DATA.mkdir(exist_ok=True)
        LIVE_FILE.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    return snap


def load_live_cached() -> dict | None:
    return json.loads(LIVE_FILE.read_text(encoding="utf-8")) if LIVE_FILE.exists() else None


# --------------------------------------------------------------------------- replay
# Replay has no archived forecast ensemble, so two stand-ins are used and labelled as such:
#  * rain: the spread across a 5x5 neighbourhood (0.1 deg spacing) in ERA5 and, from 2017, ECMWF IFS analyses;
#  * discharge: an ASSUMED lognormal forecast error (sigma_log = 0.15) around the reanalysis peak.
DISCHARGE_SIGMA_LOG = 0.15
N_DISCHARGE_MEMBERS = 31


def _lognormal_members(peak: float) -> list[float]:
    from scipy.stats import norm
    qs = (np.arange(N_DISCHARGE_MEMBERS) + 0.5) / N_DISCHARGE_MEMBERS
    return [round(float(peak * np.exp(DISCHARGE_SIGMA_LOG * norm.ppf(q))), 1) for q in qs]


def fetch_replay(event_id: str, save: bool = True) -> dict:
    ev = EVENT_BY_ID[event_id]
    ref = load_reference()
    d0 = date.fromisoformat(ev["as_of"])
    d1 = d0 + timedelta(days=WINDOW_DAYS - 1)
    models = ["era5"] + (["ecmwf_ifs"] if d0.year >= 2017 else [])
    rids, rlat, rlon = _river_points(ref)
    fl = _as_list(_get(FLOOD_URL, latitude=rlat, longitude=rlon, daily="river_discharge", start_date=str(d0), end_date=str(d1)))
    fl_by = dict(zip(rids, fl))

    snap = {"mode": "replay", "as_of": ev["as_of"], "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "event": ev, "source": f"Open-Meteo archive API ({' + '.join(models)} rain), GloFAS reanalysis discharge. Hindcast, not an archived forecast.",
            "cities": {}}
    for c in CITIES:
        pts = [(round(c["lat"] + i * 0.1, 3), round(c["lon"] + j * 0.1, 3)) for i in range(-2, 3) for j in range(-2, 3)]
        members, centre = [], None
        for m in models:
            grid = _as_list(_get(ARCHIVE_URL, latitude=",".join(str(p[0]) for p in pts), longitude=",".join(str(p[1]) for p in pts),
                                 start_date=str(d0), end_date=str(d1), daily="precipitation_sum", timezone=TZ, models=m))
            vals = [g["daily"]["precipitation_sum"] for g in grid]
            members += [round(_roll3_max(v), 2) for v in vals if any(x is not None for x in v)]
            if any(x is not None for x in vals[12]):
                centre = vals[12]            # centre of the 5x5 grid; the later (higher-resolution) model wins
                days = grid[12]["daily"]["time"]
        entry = {"days": days, "rain_daily_mm": _clean(centre),
                 "rain_members_72h_mm": members,
                 "rain_member_source": f"reanalysis neighbourhood, {' + '.join(models)} ({len(members)} points within about 20 km)",
                 "discharge_daily_m3s": None, "discharge_members_peak_m3s": None, "discharge_member_source": None}
        if c["id"] in fl_by:
            q = fl_by[c["id"]]["daily"]["river_discharge"]
            peak = max(v for v in q if v is not None)
            entry.update({"discharge_daily_m3s": _clean(q), "discharge_members_peak_m3s": _lognormal_members(peak),
                          "discharge_member_source": f"assumed lognormal forecast error, sigma_log = {DISCHARGE_SIGMA_LOG} (no archived ensemble)"})
        snap["cities"][c["id"]] = entry
    if save:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        (REPLAY_DIR / f"{event_id}.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    return snap


def load_replay(event_id: str) -> dict:
    return json.loads((REPLAY_DIR / f"{event_id}.json").read_text(encoding="utf-8"))


def available_replays() -> list[dict]:
    return [e for e in EVENTS if (REPLAY_DIR / f"{e['id']}.json").exists()]


def main():
    ap = argparse.ArgumentParser(description="Collect Open-Meteo weather and river data for Baadal")
    ap.add_argument("--build", action="store_true", help="build river reference statistics and all replay snapshots")
    ap.add_argument("--live", action="store_true", help="fetch and save the live snapshot")
    args = ap.parse_args()
    if args.build:
        print("Building river reference statistics (2000-2023)...")
        build_reference()
        for e in EVENTS:
            if (REPLAY_DIR / f"{e['id']}.json").exists():
                continue
            print(f"Fetching replay {e['id']}...", flush=True)
            fetch_replay(e["id"])
    if args.live or not args.build:
        s = fetch_live()
        print(f"Live snapshot for {s['as_of']} saved to {LIVE_FILE}")


if __name__ == "__main__":
    main()
