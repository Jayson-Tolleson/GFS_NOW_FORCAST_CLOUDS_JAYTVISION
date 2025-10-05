
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_precip_ssl_cloud_server.py

Cloud-ready GFS convective precipitation + cloud voxel server for Cesium.

Features in this variant:
 - long-lived aiohttp.ClientSession inside SSE generator (avoids "Session is closed")
 - robust read_arr() that collapses extra dims, pads/truncates safely
 - realistic cloud heuristics: cloud_top, cloud_base, thickness, center computed per voxel
 - heuristic fallback when pressure-derived height is missing/unreliable
 - convective overshoot when precipitation is strong
 - opaque, loud rain with blue→green→yellow→orange→red ramp and client-side scaling
 - hover-tooltips with all voxel metadata
 - HUD intact with legend and counters
"""
import os
import io
import math
import json
import asyncio
import logging
from datetime import datetime
from urllib.parse import urlencode
from contextlib import suppress

import numpy as np
import xarray as xr
import aiohttp

from quart import Quart, request, Response, render_template_string
from werkzeug.routing import BaseConverter

# ----------------------------
# API keys (override via env)
# ----------------------------

CESIUM_ION_TOKEN = os.environ.get("CESIUM_ION_TOKEN",
    "...")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY",
    "...")
# ----------------------------
# Server config & defaults
# ----------------------------
PORT = int(os.environ.get("PORT", 8092))
SSL_CERT = os.environ.get("SSL_CERT", "security/fullchain.pem")
SSL_KEY = os.environ.get("SSL_KEY", "security/privkey.pem")

NCSS_BASE = os.environ.get(
    "NCSS_BASE",
    "https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg/TwoD"
)

MAX_VOXELS_PER_TILE = int(os.environ.get("MAX_VOXELS_PER_TILE", "2000"))
MAX_WORKERS_LIMIT = int(os.environ.get("MAX_WORKERS_LIMIT", "24"))
MAX_TILE_BYTES = int(os.environ.get("MAX_TILE_BYTES", str(50 * 1024 * 1024)))
DEFAULT_VOXEL_BATCH = int(os.environ.get("VOXEL_BATCH_SIZE", "128"))
DEFAULT_WORKERS = int(os.environ.get("DEFAULT_WORKERS", "6"))
FETCH_SEMAPHORE = int(os.environ.get("FETCH_SEMAPHORE", "6"))

# precipitation tuning
RAIN_THRESHOLD_MM_PER_HR = float(os.environ.get("RAIN_THRESHOLD_MM_PER_HR", "1.0"))
RAIN_HEATMAP_CLAMP = float(os.environ.get("RAIN_HEATMAP_CLAMP", "10.0"))
DISPLAY_RAIN_THRESHOLD_MMHR = float(os.environ.get("DISPLAY_RAIN_THRESHOLD_MMHR", "0.0"))

FIXED_VARS = [
    "Convective_precipitation_rate_surface",
    "Low_cloud_cover_low_cloud_Mixed_intervals_Average",
    "Medium_cloud_cover_middle_cloud_Mixed_intervals_Average",
    "High_cloud_cover_high_cloud_Mixed_intervals_Average",
    "Pressure_low_cloud_top_Mixed_intervals_Average",
    "Pressure_middle_cloud_top_Mixed_intervals_Average",
    "Pressure_high_cloud_top_Mixed_intervals_Average",
    "u-component_of_wind_sigma",
    "v-component_of_wind_sigma"
]

# realistic cloud tops (meters) - adjustable via env
CLOUD_TOP_MAX_LOW_M  = float(os.environ.get("CLOUD_TOP_MAX_LOW_M", 3000.0))    # ~10,000 ft
CLOUD_TOP_MAX_MID_M  = float(os.environ.get("CLOUD_TOP_MAX_MID_M", 8000.0))    # ~26,000 ft
CLOUD_TOP_MAX_HIGH_M = float(os.environ.get("CLOUD_TOP_MAX_HIGH_M", 12200.0))  # ~40,000 ft

# fallback defaults (if pressure_to_height fails)
CLOUD_DEFAULT_TOP_LOW_M  = float(os.environ.get("CLOUD_DEFAULT_TOP_LOW_M", 1500.0))
CLOUD_DEFAULT_TOP_MID_M  = float(os.environ.get("CLOUD_DEFAULT_TOP_MID_M", 5000.0))
CLOUD_DEFAULT_TOP_HIGH_M = float(os.environ.get("CLOUD_DEFAULT_TOP_HIGH_M", 10000.0))

# typical thickness defaults by cloud type (meters)
CLOUD_THICKNESS_DEFAULT = {
    "low": 2000.0,
    "medium": 3000.0,
    "high": 1500.0
}
CLOUD_THICKNESS_MIN = float(os.environ.get("CLOUD_THICKNESS_MIN", 500.0))
CLOUD_THICKNESS_MAX = float(os.environ.get("CLOUD_THICKNESS_MAX", 8000.0))

# convective overshoot parameters (how much to extend cloud_top when precipitation is strong)
CONVECTIVE_OVERSHOOT_FACTOR = float(os.environ.get("CONVECTIVE_OVERSHOOT_FACTOR", 1.3))
CONVECTIVE_MMHR_THRESHOLD = float(os.environ.get("CONVECTIVE_MMHR_THRESHOLD", 2.0))  # mm/hr

# logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gfs_precip_ssl_cloud_server")

# app
app = Quart(__name__, static_folder=None)

class FloatConverter(BaseConverter):
    regex = r"-?\d+(?:\.\d+)?"
    def to_python(self, value): return float(value)
    def to_url(self, value): return str(value)
app.url_map.converters['float'] = FloatConverter

# ----------------------------
# helpers
# ----------------------------
def wrap180(x): return ((x + 180.0) % 360.0) - 180.0
def normalize_lon(lon):
    try: return float(((float(lon) + 180.0) % 360.0) - 180.0)
    except Exception: return None
def sanitize_number(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v): return None
        return v
    except Exception:
        return None

def pressure_to_height(p):
    """
    Convert pressure (Pa or hPa) to approximate height in meters using barometric formula.
    If p > 2000 assume Pa -> convert to hPa first.
    """
    try: pv = float(p)
    except Exception: return 0.0
    if not np.isfinite(pv) or pv <= 0: return 0.0
    p_hpa = pv/100.0 if pv > 2000.0 else pv
    return 44330.0 * (1.0 - (p_hpa / 1013.25) ** (1.0 / 5.255))

def heuristic_cloud_height(layer: str, frac: float, precip_mm_hr: float = 0.0) -> float:
    """
    Heuristic cloud top altitude in meters given layer, cloud fraction, and precipitation.
    Returns a plausible top altitude (not clamped to global caps here).
    """
    if layer == "low":
        base, top = 500.0, 3000.0
    elif layer == "medium":
        base, top = 3000.0, 8000.0
    elif layer == "high":
        base, top = 8000.0, 12000.0
    else:
        base, top = 1000.0, 5000.0

    # extend top a bit if fraction is high (dense deck tends to higher top within the layer),
    # and allow convective overshoot when precipitation strong
    frac_factor = 0.5 + frac * 0.5   # maps frac 0->0.5, 1->1.0
    top_adj = top * (1.0 + 0.15 * (frac - 0.5))  # slight nudge for high fraction
    if precip_mm_hr > CONVECTIVE_MMHR_THRESHOLD and layer in ("low", "medium"):
        top_adj *= CONVECTIVE_OVERSHOOT_FACTOR
    return base + frac_factor * (top_adj - base)

def infer_cloud_type(layer: str, frac: float, precip_mm_hr: float) -> str:
    """
    Simple heuristic to pick a cloud 'type' string for tooltip:
    - 'Cumulonimbus' when precipitation strong and low/mid layer
    - layer-based defaults otherwise
    """
    if precip_mm_hr > 2.0 and layer in ("low", "medium"):
        return "Cumulonimbus (convective)"
    if layer == "low":
        return "Cumulus / Stratocumulus"
    if layer == "medium":
        return "Altostratus / Altocumulus"
    return "Cirrus / High cloud"

def build_ncss_params(north, south, east, west, horizStride=1, vertStride=None, accept="netcdf4-classic"):
    params = [("var", v) for v in FIXED_VARS]
    params += [
        ("north", f"{north:.6f}"),
        ("south", f"{south:.6f}"),
        ("east",  f"{east:.6f}"),
        ("west",  f"{west:.6f}"),
        ("horizStride", str(int(horizStride))),
        ("time", "present"),
        ("accept", accept)
    ]
    if vertStride: params.append(("vertStride", str(int(vertStride))))
    return params

def get_tiles_in_bbox(north, south, east, west, grid_n):
    west, east = wrap180(west), wrap180(east)
    if east <= west: east += 360.0
    lats = np.linspace(south, north, grid_n + 1)
    lons = np.linspace(west, east, grid_n + 1)
    tiles = []
    tindex = 0
    for i in range(grid_n):
        tile_s, tile_n = float(lats[i]), float(lats[i+1])
        for j in range(grid_n):
            tile_w, tile_e = float(lons[j]), float(lons[j+1])
            tiles.append((tindex, i, j, tile_n, tile_s, tile_w, tile_e))
            tindex += 1
    return tiles

def _blocking_safe_open_xarray(ds_bytes):
    buf = io.BytesIO(ds_bytes)
    engines = ("h5netcdf", None)
    last_exc = None
    for engine in engines:
        try:
            buf.seek(0)
            ds = xr.open_dataset(buf, engine=engine) if engine else xr.open_dataset(buf)
            return ds
        except Exception as e:
            last_exc = e
            buf.seek(0)
            continue
    raise last_exc if last_exc else RuntimeError("safe_open_xarray: unknown error")

def stable_voxel_id(tileIndex, lat, lon, layer):
    la, lo = int(round(lat * 10000.0)), int(round(lon * 10000.0))
    return f"v_{tileIndex}_{la}_{lo}_{layer}"

# ----------------------------
# fetch & process tile
# ----------------------------
async def fetch_and_process_tile(aio_session, params, tileIndex, progress_q, horizStride=1,
                                 max_tile_bytes=MAX_TILE_BYTES, timeout=90, voxel_batch=DEFAULT_VOXEL_BATCH,
                                 czml_every=12, semaphore=None):
    url = NCSS_BASE + "?" + urlencode(params)
    await progress_q.put({"type":"debug","message":f"tile {tileIndex} start url_len={len(url)}"})
    try:
        if semaphore: await semaphore.acquire()
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aio_session.get(url, timeout=timeout_cfg) as resp:
            if resp.status != 200:
                txt = (await resp.text())[:512]
                await progress_q.put({"type":"error","tileIndex":tileIndex,"message":f"HTTP {resp.status} {txt!r}"})
                return
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64*1024):
                buf.extend(chunk)
                if len(buf) > max_tile_bytes:
                    await progress_q.put({"type":"error","tileIndex":tileIndex,"message":"exceeded max tile size"})
                    return
            ds_bytes = bytes(buf)
    except RuntimeError as e:
        await progress_q.put({"type":"error","tileIndex":tileIndex,"message":f"fetch failed: {e}"})
        return
    finally:
        if semaphore:
            try: semaphore.release()
            except: pass

    loop = asyncio.get_running_loop()
    try:
        ds = await loop.run_in_executor(None, lambda: _blocking_safe_open_xarray(ds_bytes))
    except Exception as e:
        await progress_q.put({"type":"error","tileIndex":tileIndex,"message":f"xarray open failed: {e}"})
        return

    try:
        lat_name = next((n for n in ("lat","latitude","y") if n in ds), None)
        lon_name = next((n for n in ("lon","longitude","x") if n in ds), None)
        if not lat_name or not lon_name:
            ds.close()
            await progress_q.put({"type":"error","tileIndex":tileIndex,"message":"no lat/lon vars"})
            return

        flat_lats, flat_lons = np.asarray(ds[lat_name]).ravel(), np.asarray(ds[lon_name]).ravel()
        N = flat_lats.size
        if N == 0:
            ds.close()
            await progress_q.put({"type":"processed","tileIndex":tileIndex,"count":0})
            return

        # robust reader for data arrays
        def read_arr(varname):
            if not varname or varname not in ds:
                return np.zeros(N, dtype=float)
            da = ds[varname]
            try:
                if hasattr(da, "dims"):
                    sel = {}
                    dims = list(da.dims)
                    if "time" in dims:
                        sel["time"] = 0
                    for extra in ("level","height","z","nv","sigma","siglay"):
                        if extra in dims:
                            sel[extra] = 0
                    if sel:
                        try:
                            da = da.isel(**sel)
                        except Exception:
                            pass
                arr = np.asarray(da).ravel().astype(float)
            except Exception:
                arr = np.asarray(da.values).ravel().astype(float)

            if arr.size == N:
                return arr
            if arr.size < N:
                out = np.zeros(N, dtype=float)
                out[:arr.size] = arr
                logger.debug("read_arr: padded var %s arr.size=%d -> N=%d", varname, arr.size, N)
                return out
            # arr.size > N
            if arr.size % N == 0:
                arr2 = arr.reshape(-1, N)
                logger.debug("read_arr: var %s had extra leading slices, taking last slice (%d x %d)", varname, arr2.shape[0], arr2.shape[1])
                return arr2[-1, :].astype(float)
            logger.debug("read_arr: truncating var %s arr.size=%d -> N=%d (falling back to first N values)", varname, arr.size, N)
            return arr[:N]

        # determine variables (works with varied naming)
        precip_var = next((v for v in ("Convective_precipitation_rate_surface","convective_precipitation_rate_surface","Total_precipitation_surface","precipitation_rate") if v in ds), None)
        cloud_low = read_arr("Low_cloud_cover_low_cloud_Mixed_intervals_Average")
        cloud_mid = read_arr("Medium_cloud_cover_middle_cloud_Mixed_intervals_Average")
        cloud_high = read_arr("High_cloud_cover_high_cloud_Mixed_intervals_Average")
        p_low = read_arr("Pressure_low_cloud_top_Mixed_intervals_Average")
        p_mid = read_arr("Pressure_middle_cloud_top_Mixed_intervals_Average")
        p_high = read_arr("Pressure_high_cloud_top_Mixed_intervals_Average")
        precip = read_arr(precip_var)

        num_points = min(N, MAX_VOXELS_PER_TILE)
        indices = np.linspace(0, N-1, num=num_points, dtype=int, endpoint=True)

        batch = []
        for idx in indices:
            cl, cm, ch, rawp = cloud_low[idx], cloud_mid[idx], cloud_high[idx], float(precip[idx])

            # Decide layer and compute realistic cloud top/base/thickness
            if ch > 1e-6:
                layer = "high"
                frac = float(ch)
                raw_top = pressure_to_height(p_high[idx])
                if not np.isfinite(raw_top) or raw_top <= 0:
                    cloud_top = heuristic_cloud_height("high", frac, rawp * 3600.0)
                else:
                    cloud_top = float(raw_top)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_HIGH_M))

            elif cm > 1e-6:
                layer = "medium"
                frac = float(cm)
                raw_top = pressure_to_height(p_mid[idx])
                if not np.isfinite(raw_top) or raw_top <= 0:
                    cloud_top = heuristic_cloud_height("medium", frac, rawp * 3600.0)
                else:
                    cloud_top = float(raw_top)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_MID_M))

            elif cl > 1e-6:
                layer = "low"
                frac = float(cl)
                raw_top = pressure_to_height(p_low[idx])
                if not np.isfinite(raw_top) or raw_top <= 0:
                    cloud_top = heuristic_cloud_height("low", frac, rawp * 3600.0)
                else:
                    cloud_top = float(raw_top)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_LOW_M))

            elif rawp > 0.0:
                # precipitation voxels near-surface columns
                layer, frac, cloud_top = "precip", 0.25, 2000.0
            else:
                continue

            lat_val, lon_val = sanitize_number(flat_lats[idx]), sanitize_number(normalize_lon(flat_lons[idx]))
            if lat_val is None or lon_val is None:
                continue
            vid = stable_voxel_id(tileIndex, lat_val, lon_val, layer)

            if layer == "precip":
                units = ds[precip_var].attrs.get("units", "unknown") if precip_var else "unknown"
                mm_per_hr = rawp * 3600.0
                if mm_per_hr < DISPLAY_RAIN_THRESHOLD_MMHR:
                    continue
                norm = float(min(1.0, mm_per_hr / max(1e-6, RAIN_THRESHOLD_MM_PER_HR)))
                radius = max(300.0, 300.0 + 5000.0 * norm)
                length = max(4000.0, 4000.0 + 16000.0 * norm)
                center_height = max(length / 2.0 + 50.0, cloud_top - length / 2.0)
                voxel = {
                    "id": vid,
                    "type": "voxel",
                    "layer": "precip",
                    "lat": float(lat_val),
                    "lon": float(lon_val),
                    "height": center_height,
                    "geom": "cylinder",
                    "radius": radius,
                    "length": length,
                    "precip_rate_raw": rawp,
                    "precip_mm_per_hr": mm_per_hr,
                    "precip_norm": norm,
                    "heat_val": float(min(RAIN_HEATMAP_CLAMP, mm_per_hr)),
                    "precip_units": units
                }
            else:
                # compute thickness (scale by fraction, clamp)
                base_thickness = CLOUD_THICKNESS_DEFAULT.get(layer, 2000.0)
                thickness = float(np.clip(base_thickness * (0.5 + frac), CLOUD_THICKNESS_MIN, CLOUD_THICKNESS_MAX))
                cloud_base = max(0.0, cloud_top - thickness)
                center_height = cloud_base + (thickness / 2.0)

                # cloud type inference for tooltip
                mm_per_hr = rawp * 3600.0
                cloud_type = infer_cloud_type(layer, frac, mm_per_hr)

                size_xy = max(2000.0, 6000.0 * max(0.25, frac))
                size_z = float(np.clip(thickness, 1200.0, 10000.0))
                voxel = {
                    "id": vid,
                    "type": "voxel",
                    "layer": layer,
                    "lat": float(lat_val),
                    "lon": float(lon_val),
                    "height": float(center_height),
                    "geom": "box",
                    "size_m": [size_xy, size_xy, size_z],
                    "cloud_frac": float(frac),
                    "cloud_top": float(cloud_top),
                    "cloud_base": float(cloud_base),
                    "thickness_m": float(size_z),
                    "cloud_type": cloud_type
                }

            batch.append(voxel)
            if len(batch) >= voxel_batch:
                await progress_q.put({"type":"voxels","tileIndex":tileIndex,"voxels":batch})
                batch = []

        if batch:
            await progress_q.put({"type":"voxels","tileIndex":tileIndex,"voxels":batch})
        ds.close()
        await progress_q.put({"type":"mapped","tileIndex":tileIndex})

    except Exception as e:
        logger.exception("processing tile %s failed: %s", tileIndex, e)
        await progress_q.put({"type":"error","tileIndex":tileIndex,"message":str(e)})

# ----------------------------
# SSE & endpoints (generator keeps ClientSession alive)
# ----------------------------
def format_sse(event: str | None, data_obj):
    s = json.dumps(data_obj, default=str)
    msg = ""
    if event: msg += f"event: {event}\n"
    msg += f"data: {s}\n\n"
    return msg

def forecast_time_param(forecast_hours: str):
    """Return ISO8601 string for NCSS time query."""
    if not forecast_hours:
        return "present"
    try:
        hours = int(forecast_hours)
    except:
        return "present"
    t = datetime.utcnow() + timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:00:00Z")

@app.route("/voxels_stream")
async def voxels_stream():
    north = request.args.get("north", type=float)
    south = request.args.get("south", type=float)
    east = request.args.get("east", type=float)
    west = request.args.get("west", type=float)
    grid_n = request.args.get("grid_n", 6, type=int)
    forecast_hours = request.args.get("forecast_hours", None)
    tiles = get_tiles_in_bbox(north, south, east, west, grid_n)

    iso_time = forecast_time_param(forecast_hours)
    print(f"Fetching voxels at forecast time: {iso_time}")

    # Build NCSS URL here
    ncss_params = {
        "north": north, "south": south, "east": east, "west": west,
        "time": iso_time
    }

    progress_q = asyncio.Queue()
    semaphore = asyncio.Semaphore(FETCH_SEMAPHORE)

    async def worker(tile, aio_session):
        tindex, _, _, tn, ts, tw, te = tile
        params = build_ncss_params(tn, ts, te, tw)
        await fetch_and_process_tile(aio_session, params, tindex, progress_q, semaphore=semaphore)

    async def producer(aio_session):
        tasks = [asyncio.create_task(worker(t, aio_session)) for t in tiles]
        try:
            # Wait for workers to finish fetching/processing the tiles
            await asyncio.gather(*tasks)
            logger.info("producer: workers done — entering heartbeat loop to keep SSE open")
            # Heartbeat loop -> keeps SSE alive until client disconnects (or server cancels)
            while True:
                await progress_q.put({"type": "heartbeat", "ts": datetime.utcnow().isoformat()})
                await asyncio.sleep(10)  # heartbeat interval (seconds)
        except asyncio.CancelledError:
            logger.info("producer cancelled (likely client disconnected)")
            # Cancel workers if still running
            for t in tasks:
                t.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            raise
        except Exception as e:
            logger.exception("producer encountered exception: %s", e)
        finally:
            # Ensure generator can exit: put a sentinel on the queue
            try:
                await progress_q.put(None)
            except Exception:
                logger.exception("failed to put sentinel into progress_q in finally")

    async def generator():
        # Keep a long-lived aiohttp session while SSE client is connected
        session_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=session_timeout) as aio_session:
            prod_task = asyncio.create_task(producer(aio_session))
            logger.info("voxels_stream: producer started with %d tiles", len(tiles))
            try:
                while True:
                    item = await progress_q.get()
                    # sentinel -> stop streaming
                    if item is None:
                        logger.info("voxels_stream: received sentinel, ending generator")
                        break
                    # yield SSE-formatted text
                    yield format_sse(None, item)
            except asyncio.CancelledError:
                logger.info("voxels_stream: generator cancelled (client disconnected)")
                # cancel producer so it cleans up
                if not prod_task.done():
                    prod_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await prod_task
                raise
            except GeneratorExit:
                logger.info("voxels_stream: GeneratorExit (client likely disconnected)")
                if not prod_task.done():
                    prod_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await prod_task
                raise
            finally:
                # Ensure producer is cancelled if still running
                if not prod_task.done():
                    prod_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await prod_task
                logger.info("voxels_stream: generator finished, session will close")

    return Response(generator(), content_type="text/event-stream")

# ----------------------------
# CLIENT HTML (inline) - HUD + rain visualization + hover-tooltips (cloud top/base shown)
# (HUD kept intact per request)
# ----------------------------
CLIENT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Live 3D Weather HUD — Blue-Glow Clouds + Stronger Opacity</title>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.116/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.116/Build/Cesium/Cesium.js"></script>
<style>
html, body, #cesiumContainer { width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:black; }
/* HUD styling (intact) */
.hud { position: absolute; top: 15px; left: 15px; width: 380px; background: rgba(10,10,10,0.45); border: 2px solid #0ff; border-radius: 12px; padding: 10px; font-family: 'Roboto', sans-serif; font-size: 13px; color: #0ff; text-shadow: 0 0 5px #0ff; box-shadow: 0 0 12px #0ff, inset 0 0 20px #0ff; z-index: 999; transition: all 0.3s ease; cursor: move; }
.hud.minimized { width: 50px; height: 50px; padding: 5px; overflow: hidden; }
.hud-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.hud-header span { font-weight:bold; font-size:14px; color:#0ff; text-shadow:0 0 8px #0ff; }
.hud-header button { background:#0f0; color:black; border:none; padding:4px 8px; cursor:pointer; font-weight:bold; font-family:'Roboto'; box-shadow:0 0 6px #0f0; }
#fetchClouds { display:block; margin:6px 0; width:100%; background: #0ff; color:black; border:none; padding:6px; font-weight:bold; text-shadow:0 0 6px #0ff; cursor:pointer; box-shadow:0 0 10px #0ff; border-radius:6px; }
#gridN { width:45px; margin-left:4px; background: rgba(0,255,255,0.1); border:1px solid #0ff; color:#0ff; text-shadow:0 0 3px #0ff; border-radius:4px; }
.milestone { display:flex; align-items:center; margin:6px 0; position:relative; }
.light { width:14px; height:14px; border-radius:50%; background:#0f0; margin-right:6px; animation: blink 1s infinite; box-shadow:0 0 6px #0f0; }
@keyframes blink {0%,50%,100%{opacity:1;}25%,75%{opacity:0.3;}}
.count { color:#00f; font-weight:bold; margin-left:auto; text-shadow:0 0 5px #00f; }
/* tooltip for voxels */
.voxel-tooltip { position: absolute; z-index: 2000; pointer-events: none; display: none; background: rgba(0,0,0,0.9); color: #aaffff; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(0,255,255,0.12); font-size: 12px; min-width: 220px; box-shadow: 0 0 12px rgba(0,255,255,0.08); }
/* Legend */
.legend { display:flex; align-items:center; gap:8px; margin-top:8px; user-select:none; }
.legend-gradient { height:14px; flex:1; border-radius:6px; border:1px solid rgba(255,255,255,0.06); background: linear-gradient(90deg, #0000ff 0%, #00ff00 25%, #ffff00 50%, #ffa500 75%, #ff0000 100%); }
.legend-labels { display:flex; justify-content:space-between; font-size:11px; color:#ccffff; margin-top:4px; }
/* small adjustments */
#statusLight { width:12px; height:12px; border-radius:50%; display:inline-block; background:#f00; box-shadow:0 0 6px #f00; margin-right:6px; animation: sseBlink 1s infinite; }
@keyframes sseBlink {0%,50%,100%{opacity:1;}25%,75%{opacity:0.3;}}
</style>
</head>
<body>
<div id="cesiumContainer"></div>

<!-- HUD -->
<div class="hud" id="hud">
    <div class="hud-header">
        <span>3D Weather HUD</span>
        <button id="minimizeBtn">–</button>
    </div>

    <div style="margin-bottom:4px;">
        <span id="statusLight"></span>SSE Status
    </div>

    <button id="fetchClouds">Fetch Clouds</button>
    Grid N: <input type="number" id="gridN" value="6" min="1" max="12"/>

    <!-- Forecast radios -->
    <div style="margin-top:8px; font-size:12px; color:#aaffff;">Forecast Time:</div>
    <div id="forecastOptions" style="margin-bottom:6px;">
        <label><input type="radio" name="forecast" value="now" checked> Now</label>
        <label><input type="radio" name="forecast" value="12h"> 12h</label>
        <label><input type="radio" name="forecast" value="1d"> 1d</label>
        <label><input type="radio" name="forecast" value="2d"> 2d</label>
        <label><input type="radio" name="forecast" value="3d"> 3d</label>
    </div>

    <div id="milestones"></div>

    <!-- Rain legend -->
    <div style="margin-top:8px; font-size:12px; color:#aaffff;">Rain intensity (mm/hr):</div>
    <div class="legend">
        <div class="legend-gradient" aria-hidden="true"></div>
    </div>
    <div class="legend-labels">
        <div>low</div><div>moderate</div><div>high</div>
    </div>
</div>

<!-- hover tooltip -->
<div id="voxelTooltip" class="voxel-tooltip" aria-hidden="true"></div>

<script>
Cesium.Ion.defaultAccessToken = "{{ cesium_token }}";
const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayerPicker:false, timeline:false, animation:false, geocoder:false,
    homeButton:false, sceneModePicker:false, navigationHelpButton:false,
    fullscreenButton:true,
    terrainProvider: new Cesium.EllipsoidTerrainProvider()
});
viewer.scene.globe.baseColor = Cesium.Color.BLACK;
viewer.imageryLayers.removeAll();
viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url: 'https://mt0.google.com/vt/lyrs=y,traffic,transit&x={x}&y={y}&z={z}&hl=en&key={{ google_key }}',
    tilingScheme: new Cesium.WebMercatorTilingScheme(),
    maximumLevel: 20
}));

// voxel management maps
const voxelEntities = new Map();
const glowEntities = new Map();
const entityTimestamps = new Map();

// Convert hex color string to RGB array
function hexToRgb(hex){
    const h = hex.replace("#","");
    const bigint = parseInt(h,16);
    return [(bigint>>16)&255, (bigint>>8)&255, bigint&255];
}

// Linear interpolate between two RGB colors
function lerpRgb(a,b,t){
    return [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t];
}

// Convert RGB array to Cesium.Color
function rgbToCesium(rgb, alpha=1.0){
    return new Cesium.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255, alpha);
}

// ------------------- Rain & Cloud sampling -------------------

// Sample color from rain ramp based on normalized intensity
function sampleRainColor(norm){
    const t = Math.max(0, Math.min(1, norm));
    for(let i=0;i<rainStops.length-1;i++){
        const a=rainStops[i], b=rainStops[i+1];
        if(t>=a.pos && t<=b.pos){
            const local = (t - a.pos)/(b.pos - a.pos);
            const rgb = lerpRgb(a.color, b.color, local);
            return rgbToCesium(rgb, 1.0);
        }
    }
    return rgbToCesium(rainStops[rainStops.length-1].color, 1.0);
}

// Sample cloud color & alpha based on cloud fraction
function sampleCloudColorAndAlpha(frac){
    const t = Math.max(0, Math.min(1, frac || 0));
    // color interpolation
    const white = [255,255,255];
    const mid = [220,220,220];
    const dark = [40,40,40];
    let rgb;
    if(t < 0.5){
        const local = t / 0.5;
        rgb = lerpRgb(white, mid, local);
    } else {
        const local = (t - 0.5) / 0.5;
        rgb = lerpRgb(mid, dark, local);
    }
    // alpha mapping
    const minA = 0.0;
    const maxA = 0.66;
    const expo = 0.33;
    const alpha = Math.min(1.0, minA + (maxA - minA) * Math.pow(t, expo));
    return { color: new Cesium.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255, alpha), alpha: alpha };
}

// ------------------- Voxel management -------------------

// Add or update a voxel entity (cloud or precip)
function addOrUpdateVoxel(voxel){
    const id = voxel.id || `v_${voxel.lat}_${voxel.lon}_${voxel.layer}`;
    const now = Date.now();
    entityTimestamps.set(id, now);

    let ent = viewer.entities.getById(id);

    // Precip path
    if(voxel.layer === 'precip'){
        const mmhr = +voxel.precip_mm_per_hr || ((voxel.precip_rate_raw||0) * 3600.0) || 0;
        const norm = (voxel.precip_norm !== undefined) ? +voxel.precip_norm : Math.min(1.0, mmhr / Math.max(1e-6, {{ rain_clamp }}));
        const baseColor = sampleRainColor(norm);
        const bright = new Cesium.Color(Math.min(baseColor.red*1.18,1.0), Math.min(baseColor.green*1.18,1.0), Math.min(baseColor.blue*1.18,1.0), 1.0);
        const radius = Math.max(500, (voxel.radius || 500)) * 1.9;
        const length = Math.max(4000, (voxel.length || 4000)) * 1.8;
        const pos = Cesium.Cartesian3.fromDegrees(voxel.lon, voxel.lat, voxel.height || 0);

        if(!ent){
            ent = viewer.entities.add({
                id: id,
                position: pos,
                cylinder: {
                    length: length,
                    topRadius: radius,
                    bottomRadius: radius,
                    material: new Cesium.ColorMaterialProperty(bright),
                    outline: false
                }
            });
        } else {
            ent.position = pos;
            if(ent.cylinder){
                ent.cylinder.length = length;
                ent.cylinder.topRadius = radius;
                ent.cylinder.bottomRadius = radius;
                ent.cylinder.material = new Cesium.ColorMaterialProperty(bright);
            } else {
                ent.cylinder = {
                    length, topRadius: radius, bottomRadius: radius,
                    material: new Cesium.ColorMaterialProperty(bright), outline:false
                };
            }
        }

        ent.__voxel = voxel;

        // remove any old glow for precip
        const glowId = id + "_glow";
        const oldGlow = viewer.entities.getById(glowId);
        if(oldGlow) oldGlow.show = false;
        return;
    }

    // Cloud path
    const cloudFrac = Math.max(0.0, Math.min(1.0, +voxel.cloud_frac || 0.0));
    const dims = voxel.size_m && voxel.size_m.length === 3 ? voxel.size_m : [
        Math.max(3000, 6000 * Math.max(0.25, cloudFrac)),
        Math.max(2500, 6000 * Math.max(0.25, cloudFrac)),
        Math.max(2000, 4000*(0.5 + cloudFrac))
    ];
    const pos = Cesium.Cartesian3.fromDegrees(voxel.lon, voxel.lat, voxel.height || 0);

    const ca = sampleCloudColorAndAlpha(cloudFrac);
    const colorMat = new Cesium.ColorMaterialProperty(ca.color);

    if(!ent){
        ent = viewer.entities.add({
            id: id,
            position: pos,
            box: { dimensions: new Cesium.Cartesian3(...dims), material: colorMat, outline: true,
                outlineColor: new Cesium.Color(1.0,1.0,1.0, Math.min(0.46, 0.12 + ca.alpha * 0.4)) }
        });
    } else {
        ent.position = pos;
        if(ent.box){
            ent.box.dimensions = new Cesium.Cartesian3(...dims);
            ent.box.material = colorMat;
            ent.box.outlineColor = new Cesium.Color(1.0,1.0,1.0, Math.min(0.46, 0.12 + ca.alpha * 0.4));
        } else {
            ent.box = { dimensions: new Cesium.Cartesian3(...dims), material: colorMat, outline:true,
                outlineColor: new Cesium.Color(1.0,1.0,1.0,0.28) };
        }
    }
    ent.__voxel = voxel;

    // Glow box
    const glowId = id + "_glow";
    let glow = viewer.entities.getById(glowId);
    const glowScale = 1.02;
    const glowDims = [ dims[0]*glowScale, dims[1]*glowScale, dims[2]*glowScale ];
    const glowBase = 0.5;
    const glowMax = 1;
    const glowAlpha = Math.min(glowMax, glowBase + (glowMax - glowBase) * Math.sqrt(ca.alpha));
    const glowColor = new Cesium.Color(0.0,0.9,1.0, glowAlpha);

    if(!glow){
        glow = viewer.entities.add({
            id: glowId,
            position: Cesium.Cartesian3.fromDegrees(voxel.lon, voxel.lat, voxel.height || 0),
            box: { dimensions: new Cesium.Cartesian3(...glowDims), material: new Cesium.ColorMaterialProperty(glowColor), outline: false }
        });
        glowEntities.set(glowId, glow);
    } else {
        glow.position = Cesium.Cartesian3.fromDegrees(voxel.lon, voxel.lat, voxel.height || 0);
        if(glow.box){
            glow.box.dimensions = new Cesium.Cartesian3(...glowDims);
            glow.box.material = new Cesium.ColorMaterialProperty(glowColor);
        } else {
            glow.box = { dimensions: new Cesium.Cartesian3(...glowDims), material: new Cesium.ColorMaterialProperty(glowColor), outline: false };
        }
        glow.show = true;
    }

    // Cirrus ring for high layer clouds
    if(voxel.layer === 'high'){
        const ringId = id + "_ring";
        let ring = viewer.entities.getById(ringId);
        const radiusMeters = (voxel.size_m && voxel.size_m.length>0) ? (voxel.size_m[0]*0.6) : 10000.0;
        const ringHeight = (typeof voxel.cloud_top==='number') ? voxel.cloud_top : (voxel.height||0);
        const extruded = ringHeight + Math.max(20, Math.min(500, (voxel.thickness_m||1500) * 0.05));
        const baseRgb = [230/255,245/255,255/255];

        if(!ring){
            const pulseCallback = new Cesium.CallbackProperty(function(time,result){
                const d = Cesium.JulianDate.toDate(time);
                const s = d.getTime()/1000.0;
                const alpha = 0.10 + 0.06*(0.5 + 0.5*Math.sin(s*0.9 + (voxel.lat+voxel.lon)*0.01));
                return new Cesium.Color(baseRgb[0],baseRgb[1],baseRgb[2],alpha);
            }, false);
            ring = viewer.entities.add({
                id: ringId,
                position: Cesium.Cartesian3.fromDegrees(voxel.lon,voxel.lat,ringHeight),
                ellipse: { semiMajorAxis: radiusMeters, semiMinorAxis: radiusMeters, height:ringHeight,
                    extrudedHeight: extruded, material: new Cesium.ColorMaterialProperty(pulseCallback),
                    outline:true, outlineColor: new Cesium.Color(0.9,0.95,1.0,0.45), outlineWidth:2.0,
                    classificationType: Cesium.ClassificationType.CESIUM_3D_TILE
                }
            });
        } else {
            ring.position = Cesium.Cartesian3.fromDegrees(voxel.lon,voxel.lat,ringHeight);
            if(ring.ellipse){
                ring.ellipse.semiMajorAxis = radiusMeters;
                ring.ellipse.semiMinorAxis = radiusMeters;
                ring.ellipse.height = ringHeight;
                ring.ellipse.extrudedHeight = extruded;
            }
            ring.show = true;
        }
    } else {
        const ringId = id + "_ring";
        const ring = viewer.entities.getById(ringId);
        if(ring) ring.show = false;
    }
}


// HUD Milestones & Controls
const milestonesDiv = document.getElementById("milestones");
function addMilestone(name, iconSrc, tooltipText){
    const div = document.createElement("div");
    div.className="milestone";
    div.innerHTML = `<div class="light"></div><img src="${iconSrc}" alt="icon"/><span>${name}</span><span class="count">0</span><div class="tooltip">${tooltipText}</div>`;
    milestonesDiv.appendChild(div);
    return div.querySelector(".count");
}
const cloudCount = addMilestone("Cloud Voxels","https://cdn-icons-png.flaticon.com/16/414/414927.png","Total clouds rendered");
const rainCount = addMilestone("Rain Voxels","https://cdn-icons-png.flaticon.com/16/414/414930.png","Total rain voxels");

// Minimize HUD
const hudEl = document.getElementById("hud");
document.getElementById("minimizeBtn").onclick = ()=>{ hudEl.classList.toggle("minimized"); };

// Drag HUD
let isDragging = false, offsetX=0, offsetY=0;
hudEl.addEventListener('mousedown', e=>{
    if(e.target.tagName==="BUTTON" || e.target.tagName==="INPUT") return;
    isDragging=true; offsetX=e.offsetX; offsetY=e.offsetY;
});
document.addEventListener('mouseup', ()=>{ isDragging=false; });
document.addEventListener('mousemove', e=>{
    if(isDragging){ hudEl.style.left = (e.clientX-offsetX) + "px"; hudEl.style.top = (e.clientY-offsetY) + "px"; }
});

// Hover tooltip handling
const tooltip = document.getElementById("voxelTooltip");
const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction(function(movement){
    const picked = viewer.scene.pick(movement.endPosition);
    if(picked && picked.id && picked.id.__voxel){
        const v = picked.id.__voxel;
        const lat = (v._animLat !== undefined ? v._animLat : v.lat).toFixed(4);
        const lon = (v._animLon !== undefined ? v._animLon : v.lon).toFixed(4);
        let html = `<div style="font-weight:bold;margin-bottom:4px;color:#ffffff">${(v.cloud_type||v.layer||'VOXEL').toUpperCase()}</div>`;
        html += `<div><strong>Lat, Lon:</strong> ${lat}, ${lon}</div>`;
        if(v.layer === 'precip'){
            html += `<div><strong>Center h:</strong> ${(v.height||0).toFixed(0)} m</div>`;
            html += `<div><strong>Rate:</strong> ${(v.precip_mm_per_hr||0).toFixed(2)} mm/hr</div>`;
        } else {
            html += `<div><strong>Top:</strong> ${(v.cloud_top||0).toFixed(0)} m</div>`;
            html += `<div><strong>Base:</strong> ${(v.cloud_base||0).toFixed(0)} m</div>`;
            html += `<div><strong>Cloud frac:</strong> ${(v.cloud_frac||0).toFixed(2)}</div>`;
        }
        html += `<div style="margin-top:6px;font-size:11px;color:#88ffff">voxel id: ${v.id}</div>`;
        tooltip.innerHTML = html;
        tooltip.style.display = "block";
        tooltip.style.left = (movement.endPosition.x + 14) + "px";
        tooltip.style.top = (movement.endPosition.y + 14) + "px";
    } else { tooltip.style.display = "none"; }
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
viewer.scene.canvas.addEventListener("mouseout", ()=>{ tooltip.style.display = "none"; });

// SSE connection & fetch clouds
let sseSource = null;
document.getElementById("fetchClouds").onclick = ()=>{
    const bounds = viewer.camera.computeViewRectangle(viewer.scene.globe.ellipsoid);
    if(!bounds) return;
    const north = Cesium.Math.toDegrees(bounds.north);
    const south = Cesium.Math.toDegrees(bounds.south);
    const west = Cesium.Math.toDegrees(bounds.west);
    const east = Cesium.Math.toDegrees(bounds.east);
    const grid_n = document.getElementById("gridN").value || 6;

    // forecast selection
    const forecastRadios = document.getElementsByName("forecast");
    let forecastValue = "now"; // default
    for(const r of forecastRadios){ if(r.checked){ forecastValue=r.value; break; } }

    if(sseSource){ try{sseSource.close();}catch(e){} sseSource=null; }

    cloudCount.textContent="0";
    rainCount.textContent="0";

    const url = `/voxels_stream?north=${north}&south=${south}&west=${west}&east=${east}&grid_n=${grid_n}&forecast=${forecastValue}`;
    sseSource = new EventSource(url);

    const statusLight = document.getElementById("statusLight");
    sseSource.onopen = ()=>{ statusLight.style.background="#0f0"; statusLight.style.boxShadow="0 0 6px #0f0"; };
    sseSource.onerror = ()=>{ statusLight.style.background="#f00"; statusLight.style.boxShadow="0 0 6px #f00"; };
    sseSource.onmessage = (e)=>{
        try{
            const msg = JSON.parse(e.data);
            if(msg.type==='heartbeat') return;
            if(msg.type==='voxels'){
                msg.voxels.forEach(addOrUpdateVoxel);
                msg.voxels.forEach(v=>{
                    if(v.layer==='precip') rainCount.textContent = parseInt(rainCount.textContent||"0")+1;
                    else cloudCount.textContent = parseInt(cloudCount.textContent||"0")+1;
                });
            }
        }catch(err){ console.error("Failed to parse SSE message:", err, e.data); }
    };
};

// optional cleanup interval (unchanged)
setInterval(()=>{
    const now = Date.now();
    for(const [id, ts] of entityTimestamps.entries()){
        if(now - ts > 1000*60*30){
            const ent = viewer.entities.getById(id); if(ent) viewer.entities.remove(ent);
            const glow = viewer.entities.getById(id+"_glow"); if(glow) viewer.entities.remove(glow);
            const ring = viewer.entities.getById(id+"_ring"); if(ring) viewer.entities.remove(ring);
            entityTimestamps.delete(id); voxelEntities.delete(id); glowEntities.delete(id+"_glow");
        }
    }
}, 60*1000);

</script>
</body>
</html>

"""

# index endpoints
@app.route("/", endpoint="index_root")
@app.route("/<float:lat>/<float:lon>/<float:zoom>", endpoint="index_loc")
async def index(lat=34.05, lon=-118.24, zoom=120000):
    return await render_template_string(
        CLIENT_HTML,
        cesium_token=CESIUM_ION_TOKEN,
        google_key=GOOGLE_API_KEY,
        rain_clamp=RAIN_HEATMAP_CLAMP,
        init_lat=str(lat),
        init_lon=str(lon),
        init_zoom=str(zoom),
    )

# ----------------------------
# launch
# ----------------------------
if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config
    cfg = Config()
    cfg.bind = [f"0.0.0.0:{PORT}"]
    if SSL_CERT and SSL_KEY:
        cfg.certfile = SSL_CERT
        cfg.keyfile = SSL_KEY
    asyncio.run(hypercorn.asyncio.serve(app, cfg))

