#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server (updated):
 - Earth Engine auth-from-file or ADC
 - /init_ee accepts posted ee_key_json, ee_service_account, ee_project and recreate flag
 - robust GEE tile URL extraction to support TileFetcher objects, dicts or strings
 - /sar_proxy/{z}/{x}/{y} proxy to fetch tiles server-side and add CORS headers for Cesium
 - /gee_sar_tile returns a tile URL template (RGB) built from Sentinel-1 median (VV, VH, VV/VH)
"""

import os
import io
import math
import json
import asyncio
import logging
import tempfile
import atexit
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode, unquote
from contextlib import suppress

import numpy as np
import xarray as xr
import aiohttp
from quart import Quart, request, Response, render_template_string, jsonify
from quart_cors import cors
from werkzeug.routing import BaseConverter

# optional earthengine import (may not be present)
try:
    import ee
    EE_AVAILABLE = True
except Exception:
    ee = None
    EE_AVAILABLE = False

CESIUM_ION_TOKEN = os.environ.get("CESIUM_ION_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0YjEyYjE2Ni00MjU4LTQ0YTYtODljMy0zN2Q3NzYwZjU3YjkiLCJpZCI6MzQzNjE0LCJpYXQiOjE3NTg1NTI3ODN9.5ZqSmN4FhdDuNagZTT3w3xXRFh6541hE0mDK2aUe2xQ")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY",
    "AIzaSyANdFaa8vR69qKceJKvQBC9lyG6qb4ZPlQ")
EE_SA = os.environ.get("EE_SERVICE_ACCOUNT","goes-viewer@BroadcasterFishmap.iam.gserviceaccount.com")
EE_KEY_JSON = os.environ.get("EE_KEY_JSON","broadcasterfishmap-7d8efb6c83c1.json")
EE_PROJECT = os.environ.get("EE_PROJECT","broadcasterfishmap")
PORT = int(os.environ.get("PORT", 8092))
SSL_CERT = os.environ.get("SSL_CERT", "security/fullchain.pem")
SSL_KEY = os.environ.get("SSL_KEY", "security/privkey.pem")

NCSS_BASE = os.environ.get(
    "NCSS_BASE",
    "https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg/TwoD"
)

MAX_VOXELS_PER_TILE = int(os.environ.get("MAX_VOXELS_PER_TILE", "2000"))
MAX_TILE_BYTES = int(os.environ.get("MAX_TILE_BYTES", str(50 * 1024 * 1024)))
DEFAULT_VOXEL_BATCH = int(os.environ.get("VOXEL_BATCH_SIZE", "128"))
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

# cloud top constants and others (kept from original)
CLOUD_TOP_MAX_LOW_M  = float(os.environ.get("CLOUD_TOP_MAX_LOW_M", 3000.0))
CLOUD_TOP_MAX_MID_M  = float(os.environ.get("CLOUD_TOP_MAX_MID_M", 8000.0))
CLOUD_TOP_MAX_HIGH_M = float(os.environ.get("CLOUD_TOP_MAX_HIGH_M", 12200.0))
CLOUD_THICKNESS_DEFAULT = {"low":2000.0,"medium":3000.0,"high":1500.0}
CLOUD_THICKNESS_MIN = float(os.environ.get("CLOUD_THICKNESS_MIN", 500.0))
CLOUD_THICKNESS_MAX = float(os.environ.get("CLOUD_THICKNESS_MAX", 8000.0))
CONVECTIVE_OVERSHOOT_FACTOR = float(os.environ.get("CONVECTIVE_OVERSHOOT_FACTOR", 1.3))
CONVECTIVE_MMHR_THRESHOLD = float(os.environ.get("CONVECTIVE_MMHR_THRESHOLD", "2.0"))

# logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gfs_precip_ssl_cloud_server")

# app
app = Quart(__name__, static_folder=None)
app = cors(app, allow_origin="*")  # allow all origins

class FloatConverter(BaseConverter):
    regex = r"-?\d+(?:\.\d+)?"
    def to_python(self, value): return float(value)
    def to_url(self, value): return str(value)
app.url_map.converters['float'] = FloatConverter

# ----------------------------
# Earth Engine init helper (improved; single EE_KEY_JSON usage)
# ----------------------------
_EE_TEMP_KEY_PATH = None

def _write_temp_keyfile(key_json_str: str):
    fd, path = tempfile.mkstemp(prefix="ee_key_", suffix=".json")
    try:
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        with os.fdopen(fd, "w") as f:
            f.write(key_json_str)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        logger.debug("Wrote temporary EE key file (path hidden in logs for safety).")
        return path
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise

def _cleanup_temp_key():
    global _EE_TEMP_KEY_PATH
    if _EE_TEMP_KEY_PATH and os.path.exists(_EE_TEMP_KEY_PATH):
        try:
            os.unlink(_EE_TEMP_KEY_PATH)
            logger.info("Removed temporary EE key file.")
        except Exception:
            logger.exception("Failed to remove temporary EE key file.")
    _EE_TEMP_KEY_PATH = None

atexit.register(_cleanup_temp_key)

def initialize_earth_engine(recreate_temp_keyfile: bool = False):
    global EE_AVAILABLE, ee, EE_SA, EE_KEY_JSON, EE_PROJECT, _EE_TEMP_KEY_PATH

    if not EE_AVAILABLE:
        msg = "earthengine-api library not installed"
        logger.warning(msg)
        return False, msg

    if recreate_temp_keyfile and _EE_TEMP_KEY_PATH:
        try:
            os.unlink(_EE_TEMP_KEY_PATH)
        except Exception:
            pass
        _EE_TEMP_KEY_PATH = None

    if EE_KEY_JSON:
        try:
            maybe = str(EE_KEY_JSON).strip()
            if maybe.startswith("{"):
                if not _EE_TEMP_KEY_PATH or recreate_temp_keyfile:
                    _EE_TEMP_KEY_PATH = _write_temp_keyfile(EE_KEY_JSON)
                key_file = _EE_TEMP_KEY_PATH
            else:
                key_file = EE_KEY_JSON

            if not os.path.isfile(key_file):
                msg = f"EE key file not found at: {key_file}"
                logger.error(msg)
                return False, msg

            if EE_SA:
                creds = ee.ServiceAccountCredentials(EE_SA, key_file)
                if EE_PROJECT:
                    ee.Initialize(creds, project=EE_PROJECT)
                else:
                    ee.Initialize(creds)
                msg = f"Initialized Earth Engine using service account and key file."
                logger.info(msg)
                return True, msg
            else:
                try:
                    creds = ee.ServiceAccountCredentials(None, key_file)
                    if EE_PROJECT:
                        ee.Initialize(creds, project=EE_PROJECT)
                    else:
                        ee.Initialize(creds)
                    msg = f"Initialized Earth Engine using key file (no EE_SERVICE_ACCOUNT provided)."
                    logger.info(msg)
                    return True, msg
                except Exception as ex_inner:
                    logger.exception("ServiceAccountCredentials(None, key_file) failed: %s", ex_inner)
                    try:
                        ee.Initialize()
                        msg = "Initialized Earth Engine using default credentials (after key-file attempt failed)"
                        logger.info(msg)
                        return True, msg
                    except Exception as ex_default:
                        logger.exception("Fallback ee.Initialize() failed: %s", ex_default)
                        return False, f"failed init from key file: {ex_inner}; fallback init also failed: {ex_default}"

        except Exception as e:
            logger.exception("Failed to initialize Earth Engine using EE_KEY_JSON: %s", e)
            return False, f"failed init from EE_KEY_JSON: {e}"

    try:
        ee.Initialize()
        msg = "Initialized Earth Engine using default credentials (ADC or user auth)"
        logger.info(msg)
        return True, msg
    except Exception as e:
        logger.exception("ee.Initialize() with default credentials failed: %s", e)
        return False, f"default ee.Initialize() failed: {e}"

# Attempt initialization at startup
EE_READY = False
EE_INIT_MSG = "not attempted"
try:
    ok, msg = initialize_earth_engine()
    EE_READY = bool(ok)
    EE_INIT_MSG = msg
except Exception as e:
    EE_READY = False
    EE_INIT_MSG = f"exception during init: {e}"
    logger.exception("Earth Engine initialization raised: %s", e)

logger.info("EE_READY=%s, message=%s", EE_READY, EE_INIT_MSG)

# ----------------------------
# helpers (same as before)
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
    try: pv = float(p)
    except Exception: return 0.0
    if not np.isfinite(pv) or pv <= 0: return 0.0
    p_hpa = pv/100.0 if pv > 2000.0 else pv
    return 44330.0 * (1.0 - (p_hpa / 1013.25) ** (1.0 / 5.255))
def heuristic_cloud_height(layer: str, frac: float, precip_mm_hr: float = 0.0) -> float:
    if layer == "low":
        base, top = 500.0, 3000.0
    elif layer == "medium":
        base, top = 3000.0, 8000.0
    elif layer == "high":
        base, top = 8000.0, 12000.0
    else:
        base, top = 1000.0, 5000.0
    frac_factor = 0.5 + frac * 0.5
    top_adj = top * (1.0 + 0.15 * (frac - 0.5))
    if precip_mm_hr > CONVECTIVE_MMHR_THRESHOLD and layer in ("low","medium"):
        top_adj *= CONVECTIVE_OVERSHOOT_FACTOR
    return base + frac_factor * (top_adj - base)

def infer_cloud_type(layer: str, frac: float, precip_mm_hr: float) -> str:
    if precip_mm_hr > 2.0 and layer in ("low","medium"):
        return "Cumulonimbus (convective)"
    if layer == "low": return "Cumulus / Stratocumulus"
    if layer == "medium": return "Altostratus / Altocumulus"
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
# fetch_and_process_tile (identical logic retained)
# ----------------------------
async def fetch_and_process_tile(aio_session, params, tileIndex, progress_q, generation_id,
                                 horizStride=1, max_tile_bytes=MAX_TILE_BYTES, timeout=90,
                                 voxel_batch=DEFAULT_VOXEL_BATCH, semaphore=None):
    url = NCSS_BASE + "?" + urlencode(params)
    await progress_q.put({"type":"debug","generation_id":generation_id,"message":f"tile {tileIndex} start url_len={len(url)}"})
    try:
        if semaphore: await semaphore.acquire()
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aio_session.get(url, timeout=timeout_cfg) as resp:
            if resp.status != 200:
                txt = (await resp.text())[:512]
                await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":f"HTTP {resp.status} {txt!r}"})
                return
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64*1024):
                buf.extend(chunk)
                if len(buf) > max_tile_bytes:
                    await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":"exceeded max tile size"})
                    return
            ds_bytes = bytes(buf)
    except Exception as e:
        await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":f"fetch failed: {e}"})
        return
    finally:
        if semaphore:
            try: semaphore.release()
            except: pass

    loop = asyncio.get_running_loop()
    try:
        ds = await loop.run_in_executor(None, lambda: _blocking_safe_open_xarray(ds_bytes))
    except Exception as e:
        await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":f"xarray open failed: {e}"})
        return

    try:
        lat_name = next((n for n in ("lat","latitude","y") if n in ds), None)
        lon_name = next((n for n in ("lon","longitude","x") if n in ds), None)
        if not lat_name or not lon_name:
            ds.close()
            await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":"no lat/lon vars"})
            return

        flat_lats, flat_lons = np.asarray(ds[lat_name]).ravel(), np.asarray(ds[lon_name]).ravel()
        N = flat_lats.size
        if N == 0:
            ds.close()
            await progress_q.put({"type":"processed","generation_id":generation_id,"tileIndex":tileIndex,"count":0})
            return

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
            if arr.size % N == 0:
                arr2 = arr.reshape(-1, N)
                logger.debug("read_arr: var %s had extra leading slices, taking last slice (%d x %d)", varname, arr2.shape[0], arr2.shape[1])
                return arr2[-1, :].astype(float)
            logger.debug("read_arr: truncating var %s arr.size=%d -> N=%d (falling back to first N values)", varname, arr.size, N)
            return arr[:N]

        precip_var = next((v for v in ("Convective_precipitation_rate_surface","convective_precipitation_rate_surface","Total_precipitation_surface","precipitation_rate") if v in ds), None)
        cloud_low = read_arr("Low_cloud_cover_low_cloud_Mixed_intervals_Average")
        cloud_mid = read_arr("Medium_cloud_cover_middle_cloud_Mixed_intervals_Average")
        cloud_high = read_arr("High_cloud_cover_high_cloud_Mixed_intervals_Average")
        p_low = read_arr("Pressure_low_cloud_top_Mixed_intervals_Average")
        p_mid = read_arr("Pressure_middle_cloud_top_Mixed_intervals_Average")
        p_high = read_arr("Pressure_high_cloud_top_Mixed_intervals_Average")
        precip = read_arr(precip_var)

        u_var = next((v for v in ("u-component_of_wind_sigma","u_component_of_wind_sigma","u_component_of_wind","u") if v in ds), None)
        v_var = next((v for v in ("v-component_of_wind_sigma","v_component_of_wind_sigma","v_component_of_wind","v") if v in ds), None)
        u_arr = read_arr(u_var) if u_var else np.zeros(N, dtype=float)
        v_arr = read_arr(v_var) if v_var else np.zeros(N, dtype=float)

        num_points = min(N, int(MAX_VOXELS_PER_TILE * 2))
        indices = np.linspace(0, N-1, num=num_points, dtype=int, endpoint=True)

        batch = []
        for idx in indices:
            cl, cm, ch, rawp = cloud_low[idx], cloud_mid[idx], cloud_high[idx], float(precip[idx])
            uval = float(u_arr[idx]) if idx < u_arr.size else 0.0
            vval = float(v_arr[idx]) if idx < v_arr.size else 0.0
            wind_speed = math.sqrt(uval*uval + vval*vval)
            try:
                wind_dir_deg = float((math.degrees(math.atan2(uval, vval)) + 360.0) % 360.0)
            except Exception:
                wind_dir_deg = 0.0

            if ch > 1e-6:
                layer = "high"; frac = float(ch)
                raw_top = pressure_to_height(p_high[idx])
                cloud_top = float(raw_top) if (np.isfinite(raw_top) and raw_top>0) else heuristic_cloud_height("high", frac, rawp*3600.0)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_HIGH_M))
            elif cm > 1e-6:
                layer = "medium"; frac = float(cm)
                raw_top = pressure_to_height(p_mid[idx])
                cloud_top = float(raw_top) if (np.isfinite(raw_top) and raw_top>0) else heuristic_cloud_height("medium", frac, rawp*3600.0)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_MID_M))
            elif cl > 1e-6:
                layer = "low"; frac = float(cl)
                raw_top = pressure_to_height(p_low[idx])
                cloud_top = float(raw_top) if (np.isfinite(raw_top) and raw_top>0) else heuristic_cloud_height("low", frac, rawp*3600.0)
                cloud_top = float(min(cloud_top, CLOUD_TOP_MAX_LOW_M))
            elif rawp > 0.0:
                layer, frac, cloud_top = "precip", 0.25, 2000.0
            else:
                continue

            lat_val = sanitize_number(flat_lats[idx]); lon_val = sanitize_number(normalize_lon(flat_lons[idx]))
            if lat_val is None or lon_val is None: continue
            vid = stable_voxel_id(tileIndex, lat_val, lon_val, layer)

            wind_meta = {"wind_u":uval, "wind_v":vval, "wind_speed":wind_speed, "wind_dir_deg":wind_dir_deg}

            if layer == "precip":
                units = ds[precip_var].attrs.get("units", "unknown") if precip_var else "unknown"
                mm_per_hr = rawp * 3600.0
                if mm_per_hr < DISPLAY_RAIN_THRESHOLD_MMHR: continue
                norm = float(min(1.0, mm_per_hr / max(1e-6, RAIN_THRESHOLD_MM_PER_HR)))
                radius = max(300.0, 300.0 + 5000.0 * norm)
                length = max(4000.0, 4000.0 + 16000.0 * norm)
                center_height = max(length/2.0 + 50.0, cloud_top - length/2.0)
                voxel = {"id":vid,"type":"voxel","layer":"precip","lat":float(lat_val),"lon":float(lon_val),
                         "height":center_height,"geom":"cylinder","radius":radius,"length":length,
                         "precip_rate_raw":rawp,"precip_mm_per_hr":mm_per_hr,"precip_norm":norm,
                         "heat_val":float(min(RAIN_HEATMAP_CLAMP, mm_per_hr)),"precip_units":units}
                voxel.update(wind_meta)
            else:
                base_thickness = CLOUD_THICKNESS_DEFAULT.get(layer,2000.0)
                thickness = float(np.clip(base_thickness * (0.5 + frac), CLOUD_THICKNESS_MIN, CLOUD_THICKNESS_MAX))
                cloud_base = max(0.0, cloud_top - thickness)
                center_height = cloud_base + (thickness/2.0)
                mm_per_hr = rawp * 3600.0
                cloud_type = infer_cloud_type(layer, frac, mm_per_hr)
                size_xy = max(2000.0, 6000.0 * max(0.25, frac))
                size_z = float(np.clip(thickness, 1200.0, 10000.0))
                voxel = {"id":vid,"type":"voxel","layer":layer,"lat":float(lat_val),"lon":float(lon_val),
                         "height":float(center_height),"geom":"box","size_m":[size_xy,size_xy,size_z],
                         "cloud_frac":float(frac),"cloud_top":float(cloud_top),"cloud_base":float(cloud_base),
                         "thickness_m":float(size_z),"cloud_type":cloud_type}
                voxel.update(wind_meta)

            batch.append(voxel)
            if len(batch) >= voxel_batch:
                await progress_q.put({"type":"voxels","generation_id":generation_id,"tileIndex":tileIndex,"voxels":batch})
                batch = []

        if batch:
            await progress_q.put({"type":"voxels","generation_id":generation_id,"tileIndex":tileIndex,"voxels":batch})
        ds.close()
        await progress_q.put({"type":"mapped","generation_id":generation_id,"tileIndex":tileIndex})

    except Exception as e:
        logger.exception("processing tile %s failed: %s", tileIndex, e)
        await progress_q.put({"type":"error","generation_id":generation_id,"tileIndex":tileIndex,"message":str(e)})

# ----------------------------
# SSE & endpoints
# ----------------------------
def format_sse(event: str | None, data_obj):
    s = json.dumps(data_obj, default=str)
    msg = ""
    if event: msg += f"event: {event}\n"
    msg += f"data: {s}\n\n"
    return msg

def forecast_time_param(forecast_str: str):
    if not forecast_str: return "present"
    forecast_str = str(forecast_str).strip().lower()
    if forecast_str in ("now","present"): return "present"
    try:
        if forecast_str.endswith("h"):
            hours = int(forecast_str[:-1])
            t = datetime.utcnow() + timedelta(hours=hours)
            return t.strftime("%Y-%m-%dT%H:00:00Z")
        if forecast_str.endswith("d"):
            days = int(forecast_str[:-1])
            t = datetime.utcnow() + timedelta(days=days)
            return t.strftime("%Y-%m-%dT%H:00:00Z")
        hours = int(forecast_str)
        t = datetime.utcnow() + timedelta(hours=hours)
        return t.strftime("%Y-%m-%dT%H:00:00Z")
    except Exception:
        return "present"

@app.route("/voxels_stream")
async def voxels_stream():
    north = request.args.get("north", type=float)
    south = request.args.get("south", type=float)
    east = request.args.get("east", type=float)
    west = request.args.get("west", type=float)
    grid_n = request.args.get("grid_n", 6, type=int)
    forecast_param = request.args.get("forecast", "now")
    tiles = get_tiles_in_bbox(north, south, east, west, grid_n)

    iso_time = forecast_time_param(forecast_param)
    logger.info("Fetching voxels at forecast time: %s (forecast param=%r)", iso_time, forecast_param)
    generation_id = f"gen_{int(datetime.utcnow().timestamp())}"

    progress_q = asyncio.Queue()
    semaphore = asyncio.Semaphore(FETCH_SEMAPHORE)

    async def worker(tile, aio_session):
        tindex, _, _, tn, ts, tw, te = tile
        params = build_ncss_params(tn, ts, te, tw)
        params = [(k,v) for (k,v) in params if k != "time"] + [("time", iso_time)]
        await fetch_and_process_tile(aio_session, params, tindex, progress_q, generation_id, semaphore=semaphore)

    async def producer(aio_session):
        try:
            await progress_q.put({"type":"init_gen","generation_id":generation_id,"iso_time":iso_time})
        except Exception:
            logger.exception("failed to put init_gen into queue")
        tasks = [asyncio.create_task(worker(t, aio_session)) for t in tiles]
        try:
            await asyncio.gather(*tasks)
            # Keep the connection alive with heartbeats until client disconnect
            while True:
                await progress_q.put({"type":"heartbeat","generation_id":generation_id,"ts":datetime.utcnow().isoformat()})
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            raise
        except Exception as e:
            logger.exception("producer encountered exception: %s", e)
        finally:
            try:
                await progress_q.put(None)
            except Exception:
                logger.exception("failed to put sentinel into progress_q in finally")

    async def generator():
        session_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=session_timeout) as aio_session:
            prod_task = asyncio.create_task(producer(aio_session))
            try:
                while True:
                    item = await progress_q.get()
                    if item is None:
                        break
                    yield format_sse(None, item)
            except asyncio.CancelledError:
                if not prod_task.done():
                    prod_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await prod_task
                raise
            finally:
                if not prod_task.done():
                    prod_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await prod_task

    return Response(generator(), content_type="text/event-stream")


# ----------------------------
# Robust tile URL extraction for GEE map outputs
# (keeps your previous robust implementation)
def _extract_tile_url_from_mapid(mapid):
    if not mapid:
        return None
    if isinstance(mapid, str):
        return mapid
    if isinstance(mapid, dict):
        for k in ("tile_fetcher", "tileFetcher", "url", "url_format", "tiles", "mapid"):
            if k in mapid and mapid[k]:
                tf = mapid[k]
                if isinstance(tf, str):
                    return tf
                if isinstance(tf, dict):
                    for nk in ("url_format", "url", "tiles", "tile_url"):
                        v = tf.get(nk)
                        if isinstance(v, str) and v:
                            return v
                obj = tf
                for attr in ("url_format", "url", "getTileUrl", "get_tile_url", "getUrl", "get_url"):
                    if hasattr(obj, attr):
                        try:
                            val = getattr(obj, attr)
                            if callable(val):
                                try:
                                    res = val()
                                except TypeError:
                                    try:
                                        res = val({})
                                    except Exception:
                                        res = None
                                if isinstance(res, str) and res:
                                    return res
                            else:
                                if isinstance(val, str) and val:
                                    return val
                        except Exception:
                            continue
        for fallback in ("url", "url_format", "mapid"):
            v = mapid.get(fallback)
            if isinstance(v, str) and v:
                return v
        return None

    for attr in ("url_format", "url", "getTileUrl", "get_tile_url", "getUrl", "get_url", "tile_fetcher"):
        if hasattr(mapid, attr):
            try:
                val = getattr(mapid, attr)
                if callable(val):
                    try:
                        ret = val()
                    except TypeError:
                        try:
                            ret = val({})
                        except Exception:
                            ret = None
                    if isinstance(ret, str) and ret:
                        return ret
                else:
                    if isinstance(val, str) and val:
                        return val
            except Exception:
                continue
    return None

# ----------------------------
# New: gee_sar_tile endpoint (less-orange visualization, higher-res tiles)
# ----------------------------
@app.route("/gee_sar_tile")
async def gee_sar_tile():
    """
    Returns JSON with 'tile_url' (a URL template) and metadata.
    Produces a median Sentinel-1 RGB where:
      R ~ lowered VV, G ~ VH, B ~ VV/VH (ratio) with gentle non-linear stretch,
    and requests tileSize=512 for higher-res tiles from Earth Engine.
    """
    if not EE_AVAILABLE:
        return jsonify({"error": "earthengine-api not installed", "ee_ready": False}), 503
    if not EE_READY:
        return jsonify({"error": "Earth Engine not available / not initialized", "ee_ready": False, "ee_msg": EE_INIT_MSG}), 503

    try:
        north = float(request.args.get("north"))
        south = float(request.args.get("south"))
        east  = float(request.args.get("east"))
        west  = float(request.args.get("west"))
    except Exception:
        return jsonify({"error": "missing bbox params (north,south,east,west)"}), 400

    start = request.args.get("start", (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"))
    end   = request.args.get("end", datetime.utcnow().strftime("%Y-%m-%d"))

    def blocking_gee():
        try:
            geom = ee.Geometry.Rectangle([west, south, east, north])
            col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(geom)
                   .filterDate(start, end)
                   .filter(ee.Filter.eq('instrumentMode', 'IW')))

            first = col.first()
            if first is None:
                return {"tile_url": None, "error": "no_images", "message": "no Sentinel-1 images found for the requested bbox/date range"}

            # figure out band names robustly
            try:
                band_names = first.bandNames().getInfo()
            except Exception:
                band_names = col.median().bandNames().getInfo()

            vv_name = None
            vh_name = None
            for b in band_names:
                bl = b.lower()
                if 'vv' in bl and vv_name is None:
                    vv_name = b
                if 'vh' in bl and vh_name is None:
                    vh_name = b

            if not vv_name:
                return {"tile_url": None, "error": "no_vv_band", "message": f"no VV band in Sentinel-1 images; available bands: {band_names}"}

            median = col.median()

            # select & unify names
            vv = median.select([vv_name]).rename(["VV"])
            if vh_name:
                vh = median.select([vh_name]).rename(["VH"])
            else:
                vh = vv.rename(["VH"])

            # safe ratio (VV/VH)
            ratio = vv.divide(vh.add(1e-6)).rename(["RATIO"])

            # --- Visual adjustments to reduce orange and favor cyan/blues ---
            # Narrow VV red response a bit (less aggressive red), boost green (VH),
            # and emphasize ratio in blue channel. Apply mild exponents for perceptual stretch.

            # unitScale ranges are heuristics; adjust on your imagery if needed.
            vv_s = vv.unitScale(-25, -4).pow(0.85).rename(["R"])   # tighter VV -> reduce red dominance
            vh_s = vh.unitScale(-25, 0).pow(0.9).rename(["G"])     # VH -> green
            ratio_clamped = ratio.expression("min(max(x,0),6)", {"x": ratio}).rename(["RATIO"])
            ratio_s = ratio_clamped.unitScale(0, 6).pow(0.8).rename(["B"])

            # Combine with a gentle color matrix that downweights R and upweights B
            # final R = 0.55*vv + 0.05*ratio (keeps R but less orange)
            # final G = 1.0*vh
            # final B = 0.9*ratio + 0.25*vh (boost blue plus some VH)
            R = vv_s.multiply(0.55).add(ratio_s.multiply(0.05)).rename("R")
            G = vh_s.multiply(1.00).rename("G")
            B = ratio_s.multiply(0.90).add(vh_s.multiply(0.25)).rename("B")

            rgb = ee.Image.cat([R, G, B])
            # normalize to 0-1 then scale to 0-255 and uint8
            rgb = rgb.unitScale(0, 1).multiply(255).uint8()

            # request tileSize 512 for higher-res tiles, and explicitly request PNG format
            mapid = rgb.getMapId({"tileSize": 512, "format": "png"})
            tile_url = _extract_tile_url_from_mapid(mapid)
            if not tile_url:
                return {"tile_url": None, "raw_mapid": mapid, "bands": band_names}

            return {"tile_url": tile_url, "start": start, "end": end, "bands": band_names, "tileSize": 512}
        except ee.EEException as e:
            logger.exception("gee_sar_tile EEException: %s", e)
            msg = str(e)
            if 'permission' in msg.lower() or 'earthengine' in msg.lower():
                hint = ("Permission or map-creation error. Ensure your service account/project "
                        "has Earth Engine map tile creation rights and the Earth Engine API is enabled.")
                return {'tile_url': None, 'error': 'permission_denied', 'message': hint, 'raw': msg}
            return {'tile_url': None, 'error': 'ee_error', 'message': msg}
        except Exception as e:
            logger.exception("gee_sar_tile internal error: %s", e)
            return {'tile_url': None, 'error': 'internal', 'message': str(e)}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_gee)

    if not result:
        return jsonify({"error": "unknown", "message": "gee_sar_tile returned no result"}), 500
    if not result.get("tile_url"):
        return jsonify(result), 500
    return jsonify(result)
# ----------------------------
# SAR proxy endpoint (unchanged)
# ----------------------------
@app.route("/sar_proxy/<int:z>/<int:x>/<int:y>", methods=["GET", "OPTIONS"])
async def sar_proxy(z, x, y):
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    tpl_in = request.args.get("t", "")
    if not tpl_in:
        return jsonify({"error": "missing required query parameter 't' (tile URL template)"}), 400

    tpl = unquote(tpl_in)

    def fill_template(tpl_str, xx, yy, zz):
        s = re.sub(r'\{ *[xX] *\}', str(xx), tpl_str)
        s = re.sub(r'\{ *[yY] *\}', str(yy), s)
        s = re.sub(r'\{ *[zZ] *\}', str(zz), s)
        return s

    candidates = []
    if not re.search(r'\{\s*[xXyYzZ]\s*\}', tpl):
        candidates.append(tpl)
    candidates.append(fill_template(tpl, x, y, z))
    tms_y = (1 << z) - 1 - y
    candidates.append(fill_template(tpl, x, tms_y, z))

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen:
            unique_candidates.append(c); seen.add(c)

    timeout = aiohttp.ClientTimeout(total=15)
    last_err = None
    last_status = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for c in unique_candidates:
            try:
                c_try = c.strip().strip('"').strip("'")
                logger.debug("sar_proxy: trying candidate: %s", c_try)
                async with session.get(c_try) as upstream:
                    last_status = upstream.status
                    if upstream.status != 200:
                        logger.debug("sar_proxy upstream non-200 status %s for %s", upstream.status, c_try)
                        try:
                            txt = (await upstream.text())[:512]
                            logger.debug("sar_proxy upstream body (truncated): %s", txt)
                        except Exception:
                            pass
                        continue
                    content_type = upstream.headers.get("content-type", "application/octet-stream")
                    if "image" not in content_type and "application/octet-stream" != content_type:
                        logger.debug("sar_proxy upstream returned content-type=%s for %s; skipping", content_type, c_try)
                        try:
                            txt = (await upstream.text())[:512]
                            logger.debug("sar_proxy upstream body (truncated): %s", txt)
                        except Exception:
                            pass
                        last_err = f"non-image content-type '{content_type}'"
                        continue
                    body = await upstream.read()
                    resp = Response(body, status=200, content_type=content_type)
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    resp.headers["Cache-Control"] = "public, max-age=300"
                    if upstream.headers.get("etag"):
                        resp.headers["ETag"] = upstream.headers.get("etag")
                    return resp
            except Exception as e:
                last_err = str(e)
                logger.warning("sar_proxy candidate failed: %s -> %s", c, last_err)

    err_info = {
        "error": "failed to fetch any SAR tile candidate",
        "last_status": last_status,
        "last_error": last_err,
        "candidates": unique_candidates
    }
    logger.error("sar_proxy failed: %s", err_info)
    return jsonify(err_info), 502

# ----------------------------
# runtime init endpoint for EE (callable from client/admin)
# (keeps your existing init_ee implementation)
@app.route("/init_ee", methods=["POST", "GET"])
async def init_ee_endpoint():
    global EE_READY, EE_INIT_MSG, EE_KEY_JSON, EE_SA, EE_PROJECT, _EE_TEMP_KEY_PATH

    if not EE_AVAILABLE:
        return jsonify({"success": False, "message": "earthengine-api not installed on server"}), 503

    recreate_q = request.args.get("recreate_temp", None)
    recreate_body = False
    try:
        body = await request.get_json(silent=True) or {}
    except Exception:
        body = {}
    try:
        form = await request.form
        for k, v in form.items():
            if k not in body:
                body[k] = v
    except Exception:
        pass

    new_key = body.get("ee_key_json")
    new_sa  = body.get("ee_service_account")
    new_proj= body.get("ee_project")
    recreate_body = str(body.get("recreate_temp", "")).lower() in ("1","true","yes")

    if new_key:
        EE_KEY_JSON = new_key
        if isinstance(new_key, str) and new_key.strip().startswith("{"):
            if _EE_TEMP_KEY_PATH:
                try:
                    os.unlink(_EE_TEMP_KEY_PATH)
                except Exception:
                    pass
                _EE_TEMP_KEY_PATH = None
            recreate_body = True

    if new_sa:
        EE_SA = new_sa
    if new_proj:
        EE_PROJECT = new_proj

    recreate = (str(recreate_q).lower() in ("1","true","yes")) or recreate_body

    loop = asyncio.get_event_loop()

    def blocking_init():
        try:
            ok, msg = initialize_earth_engine(recreate_temp_keyfile=recreate)
            return bool(ok), str(msg)
        except Exception as e:
            logger.exception("init_ee blocking raised: %s", e)
            return False, str(e)

    try:
        ok, msg = await loop.run_in_executor(None, blocking_init)
        EE_READY = bool(ok)
        EE_INIT_MSG = msg
        return jsonify({"success": EE_READY, "message": EE_INIT_MSG})
    except Exception as e:
        logger.exception("init_ee failed: %s", e)
        return jsonify({"success": False, "message": str(e)}), 500

# ----------------------------
# Minimal index (keeps client rendering separate)
CLIENT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Live 3D Weather HUD — Blue-Glow Clouds + World Terrain + Google Imagery + SAR</title>

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
/* compact SAR controls row */
.sar-row { display:flex; align-items:center; gap:8px; margin-top:8px; }
.sar-row label{ color:#aaffff; font-size:12px; min-width:72px; }
.sar-row input[type="range"]{ width:140px; }

/* cloud opacity control */
.cloud-opacity-row { display:flex; align-items:center; gap:8px; margin-top:8px; }
.cloud-opacity-row label{ color:#aaffff; min-width:84px; }
.cloud-opacity-row input[type="range"]{ width:140px; }
</style>
</head>
<body>
<div id="cesiumContainer" aria-label="Cesium 3D view"></div>

<!-- HUD -->
<div class="hud" id="hud" role="region" aria-label="Weather HUD">
  <div class="hud-header">
    <span>3D Weather HUD</span>
    <button id="minimizeBtn" aria-label="Minimize HUD">–</button>
  </div>

  <!-- Fetch + Controls -->
  <button id="fetchClouds">Fetch Clouds</button>
  <div style="margin-top:6px;">
    <div style="display:flex; align-items:center; gap:8px;">
      <div>Grid N:</div><input id="gridN" type="number" value="6" min="1" max="18" />
      <div style="margin-left:auto; display:flex; align-items:center;">
        <button id="sarBtn" class="btn-soft" style="padding:6px;border-radius:6px;">SAR</button>
      </div>
    </div>
  </div>

  <div style="margin-top:8px;">
    <label style="margin-right:6px;"><input type="radio" name="forecast" value="now" checked> Now</label>
    <label style="margin-right:6px;"><input type="radio" name="forecast" value="12h"> +12h</label>
    <label><input type="radio" name="forecast" value="24h"> +24h</label>
  </div>

  <div style="margin-top:8px; font-size:13px;">
    <span id="statusLight" aria-hidden="true"></span><span style="vertical-align:middle; color:#ccffff;">SSE Status</span>
  </div>

  <!-- Milestones (restored) -->
  <div id="milestones" style="margin-top:8px;">
      <div class="milestone">
          <div class="light"></div>
          <img src="https://cdn-icons-png.flaticon.com/16/414/414927.png" alt="cloud icon" />
          <span style="margin-left:6px;color:#aaffff">Cloud Voxels</span>
          <span class="count" id="cloudCount">0</span>
      </div>
      <div class="milestone">
          <div class="light"></div>
          <img src="https://cdn-icons-png.flaticon.com/16/414/414930.png" alt="rain icon" />
          <span style="margin-left:6px;color:#aaffff">Rain Voxels</span>
          <span class="count" id="rainCount">0</span>
      </div>
  </div>

  <!-- Rain legend (restored) -->
  <div style="margin-top:8px; font-size:12px; color:#aaffff;">Rain intensity (mm/hr):</div>
  <div class="legend">
      <div class="legend-gradient" aria-hidden="true"></div>
  </div>
  <div class="legend-labels">
      <div>low</div><div>moderate</div><div>high</div>
  </div>

  <!-- SAR opacity control (kept but preview/debug removed) -->
  <div class="sar-row">
    <label for="sarOpacity">SAR opacity</label>
    <input id="sarOpacity" type="range" min="0" max="1" step="0.05" value="1.0"/>
    <span id="sarOpacityVal" style="color:#aaffff">1.00</span>
    <div style="margin-left:auto;">
      <button id="flyMeBtn" class="btn-soft">Fly to Me</button>
      <button id="trueNorthBtn" class="btn-soft">True North</button>
    </div>
  </div>

  <!-- Cloud opacity control (new) -->
  <div class="cloud-opacity-row">
    <label for="cloudOpacity">Cloud opacity</label>
    <input id="cloudOpacity" type="range" min="0" max="1" step="0.01" value="0.55"/>
    <div id="cloudOpacityVal" style="min-width:40px;text-align:center;color:#aaffff">0.55</div>
  </div>
</div>

<div class="voxel-tooltip" id="voxelTooltip" aria-hidden="true"></div>

<script>
/* Server templated values (replace server-side) */
Cesium.Ion.defaultAccessToken = "{{ cesium_token }}";
const GOOGLE_KEY = "{{ google_key }}";
const RAIN_CLAMP = {{ rain_clamp | float }};
const INIT_LAT = {{ init_lat }};
const INIT_LON = {{ init_lon }};
const INIT_ZOOM = {{ init_zoom }};

let viewer = null;

/* Cesium init (defensive) */
(async function initCesium(){
  try{
    let terrainProvider = new Cesium.EllipsoidTerrainProvider();
    try { terrainProvider = Cesium.createWorldTerrain(); } catch(e){ /* fallback */ }

    viewer = new Cesium.Viewer('cesiumContainer', {
      terrainProvider: terrainProvider,
      baseLayerPicker: false, timeline: false, animation: false, geocoder: false,
      homeButton: false, sceneModePicker: false, navigationHelpButton: false,
      fullscreenButton: true
    });

    viewer.scene.globe.baseColor = Cesium.Color.BLACK;
    viewer.scene.globe.depthTestAgainstTerrain = true;

    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(INIT_LON || -118.24, INIT_LAT || 34.05, INIT_ZOOM || 480000)
    });

    // optional Google imagery (falls back gracefully)
    const googleUrl = (GOOGLE_KEY && GOOGLE_KEY.length>8)
      ? `https://mt1.google.com/vt/lyrs=y,traffic,transit&x={x}&y={y}&z={z}&key=${GOOGLE_KEY}`
      : `https://mt1.google.com/vt/lyrs=y,traffic,transit&x={x}&y={y}&z={z}`;

    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
      url: googleUrl,
      credit: 'Google'
    }));

    window.viewer = viewer;
    console.info('Cesium initialized');
  } catch(err){
    console.error('Cesium init failed', err);
    const el = document.createElement('div'); el.textContent = 'Cesium init error: ' + (err && err.message? err.message:err); document.body.appendChild(el);
  }
})();

/* HUD drag & minimize */
const hud = document.getElementById('hud');
let dragOffset = null;
document.getElementById('minimizeBtn').addEventListener('click', ()=> hud.classList.toggle('minimized'));
hud.addEventListener('mousedown', e=>{
  if(e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
    dragOffset = { x: e.clientX - hud.offsetLeft, y: e.clientY - hud.offsetTop };
  }
});
document.addEventListener('mouseup', ()=> dragOffset = null);
document.addEventListener('mousemove', e=>{
  if(dragOffset){ hud.style.left = (e.clientX - dragOffset.x) + 'px'; hud.style.top  = (e.clientY - dragOffset.y) + 'px'; }
});

/* Milestones counters (restored) */
const cloudCountEl = document.getElementById('cloudCount');
const rainCountEl = document.getElementById('rainCount');
function setCounts(cloudN, rainN){
  if(cloudCountEl) cloudCountEl.textContent = String(cloudN);
  if(rainCountEl) rainCountEl.textContent = String(rainN);
}

/* SAR status helper (debugging removed from HUD) */
const sarBtn = document.getElementById('sarBtn');
const sarStatusText = { set(text){ console.info('SAR status:', text); } }; // minimal: console log only
function setSarStateText(txt){ sarStatusText.set(txt); }

/* strip-down testTileUrl (no preview DOM updates) */
async function testTileUrl(url){
  try {
    console.info('Testing tile URL (client fetch):', url);
    const resp = await fetch(url, { method: 'GET', mode: 'cors', cache: 'no-store' });
    if(!resp.ok){ console.warn('Tile fetch HTTP', resp.status); return { ok:false, status: resp.status }; }
    const ct = resp.headers.get('content-type') || '';
    if(ct.indexOf('image') === -1){
      console.warn('Non-image tile content-type', ct);
      return { ok:false, contentType: ct };
    }
    const blob = await resp.blob();
    console.info('Tile fetch OK', ct, blob.size);
    return { ok:true, contentType: ct, size: blob.size };
  } catch(err){
    console.warn('Tile fetch failed (likely CORS/network):', err && err.message ? err.message : err);
    return { ok:false, error: err && err.message ? err.message : String(err) };
  }
}

/* SAR provider helpers (kept, but HUD debugging removed) */
function addSarImageryProvider(templateUrl, options = {}) {
  const opts = Object.assign({ url: templateUrl, maximumLevel: 19, credit: 'Sentinel-1 / GEE' }, options);
  const provider = new Cesium.UrlTemplateImageryProvider(opts);
  const layer = viewer.imageryLayers.addImageryProvider(provider);
  return layer;
}

/* Toggle SAR (no preview/debug UI changes) */
let sarLayer = null;
let sarActive = false;
let sarLoading = false;
let sarTileErrors = 0;

async function toggleSAR({useProxyIfProviderErrors=true} = {}){/* same logic as before (unchanged except trimmed debug) */
  if(sarLoading){ console.warn('SAR already loading; ignoring click'); return; }
  if(sarActive){
    sarActive = false;
    if(sarLayer){ try{ viewer.imageryLayers.remove(sarLayer); } catch(e){ console.warn(e); } sarLayer = null; }
    sarBtn.classList.remove('active');
    sarBtn.textContent = 'SAR';
    setSarStateText('off');
    return;
  }
  if(!viewer || !viewer.scene || !viewer.scene.globe){ alert('Viewer not ready — try again later.'); return; }
  const bounds = viewer.camera.computeViewRectangle(viewer.scene.globe.ellipsoid);
  if(!bounds){ alert('Zoom/adjust view to request SAR.'); return; }

  sarLoading = true;
  sarBtn.disabled = true;
  sarBtn.textContent = 'SAR (loading...)';
  setSarStateText('requesting template');

  try {
    const north = Cesium.Math.toDegrees(bounds.north), south = Cesium.Math.toDegrees(bounds.south),
          west = Cesium.Math.toDegrees(bounds.west), east = Cesium.Math.toDegrees(bounds.east);
    const now = new Date(); const end = now.toISOString().slice(0,10);
    const start = new Date(now - 3*24*3600*1000).toISOString().slice(0,10);
    const url = `/gee_sar_tile?north=${north}&south=${south}&west=${west}&east=${east}&start=${start}&end=${end}`;
    console.info('Fetching SAR template:', url);
    const res = await fetch(url);
    const js = await res.json().catch(()=>null);
    if(!res.ok || !js || !js.tile_url) throw new Error(js && js.error ? js.error : 'invalid response');

    const tileTemplate = js.tile_url;
    console.info('Server template:', tileTemplate);

    function lonLatToTileXY(lon, lat, z){
      const x = Math.floor((lon + 180) / 360 * Math.pow(2, z));
      const latRad = lat * Math.PI / 180;
      const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1/Math.cos(latRad)) / Math.PI) / 2 * Math.pow(2, z));
      return { x, y, z };
    }
    const centerLon = (west + east)/2.0;
    let centerLat = (south + north)/2.0; centerLat = Math.max(Math.min(centerLat, 85.0), -85.0);
    const sampleZ = 6;
    const sample = lonLatToTileXY(centerLon, centerLat, sampleZ);
    const tmsY = Math.pow(2, sample.z) - 1 - sample.y;
    const makeCandidate = (tpl, x, y, z) => tpl.replace(/\{ *[xX] *\}/g, x).replace(/\{ *[yY] *\}/g, y).replace(/\{ *[zZ] *\}/g, z);
    const candidates = [ makeCandidate(tileTemplate, sample.x, sample.y, sample.z), makeCandidate(tileTemplate, sample.x, tmsY, sample.z) ];

    let working = null;
    for(const c of [...new Set(candidates)]) {
      const t = await testTileUrl(c);
      if(t.ok){ working = c; break; }
    }

    let finalTemplate = tileTemplate;
    let usedProxy = false;
    if(!working){
      usedProxy = true;
      finalTemplate = `/sar_proxy/{z}/{x}/{y}?t=${encodeURIComponent(tileTemplate)}`;
      console.warn('Direct tile tests failed, switching to server proxy');
    } else {
      console.info('Direct tile tests succeeded, using template');
    }

    if(sarLayer) try{ viewer.imageryLayers.remove(sarLayer); } catch(e){}
    const providerOpts = { url: finalTemplate, maximumLevel: 19, credit: 'Sentinel-1 / GEE' };
    try { providerOpts.tilingScheme = new Cesium.WebMercatorTilingScheme(); providerOpts.rectangle = Cesium.Rectangle.fromDegrees(west, south, east, north); } catch(e){}
    sarTileErrors = 0;
    sarLayer = addSarImageryProvider(finalTemplate, providerOpts);

    const alpha = parseFloat(document.getElementById('sarOpacity').value || 1.0);
    sarLayer.alpha = alpha;
    sarLayer.show = true;
    try { viewer.imageryLayers.raiseToTop(sarLayer); } catch(e){ console.warn('raiseToTop failed', e); }
    try { viewer.scene.requestRender(); } catch(e){}
    sarActive = true;
    sarBtn.classList.add('active');
    sarBtn.textContent = 'SAR (on)';
    setSarStateText(usedProxy ? 'SAR on (proxy)' : 'SAR on');

    if(!usedProxy && useProxyIfProviderErrors){
      setTimeout(()=> {
        if(sarTileErrors >= 4){
          console.warn(`Detected ${sarTileErrors} tile errors; attempting proxy fallback`);
          (async ()=>{
            try {
              if(sarLayer) try{ viewer.imageryLayers.remove(sarLayer); } catch(e){}
              const proxyTpl = `/sar_proxy/{z}/{x}/{y}?t=${encodeURIComponent(tileTemplate)}`;
              sarLayer = addSarImageryProvider(proxyTpl, providerOpts);
              sarLayer.alpha = parseFloat(document.getElementById('sarOpacity').value||1.0);
              try{ viewer.imageryLayers.raiseToTop(sarLayer); } catch(e){}
              setSarStateText('SAR on (proxy)');
            } catch(e){ console.error('proxy fallback failed', e); }
          })();
        }
      }, 3000);
    }

  } catch(err){
    console.error('toggleSAR error:', err && err.message ? err.message : err);
    setSarStateText('error');
    alert('SAR error: ' + (err && err.message ? err.message : String(err)));
    if(sarLayer) try{ viewer.imageryLayers.remove(sarLayer); } catch(e){}
    sarActive = false;
    sarBtn.classList.remove('active');
    sarBtn.textContent = 'SAR';
  } finally {
    sarLoading = false;
    sarBtn.disabled = false;
  }
}
if(sarBtn) sarBtn.addEventListener('click', ()=> toggleSAR({useProxyIfProviderErrors:true}));
document.getElementById('sarOpacity').addEventListener('input', ()=>{
  const v = parseFloat(document.getElementById('sarOpacity').value||1.0);
  document.getElementById('sarOpacityVal').textContent = v.toFixed(2);
  if(sarLayer) { sarLayer.alpha = v; try{ viewer.scene.requestRender(); } catch(e){} }
});

/* Fly-to current location & True-North */
const flyMeBtn = document.getElementById('flyMeBtn');
const trueNorthBtn = document.getElementById('trueNorthBtn');

async function flyToMyLocation(){
  if(!navigator.geolocation){ alert('Geolocation not available in this browser.'); return; }
  flyMeBtn.disabled = true;
  try{
    const pos = await new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: true, maximumAge: 60000, timeout: 15000 }));
    const lat = pos.coords.latitude, lon = pos.coords.longitude;
    const height = Math.max(1500, pos.coords.altitude || 8000);
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, Math.max(800, height)),
      orientation: { heading: viewer.camera.heading || 0.0, pitch: viewer.camera.pitch || Cesium.Math.toRadians(-45), roll: 0.0 },
      duration: 2.0
    });
  } catch(err){
    alert('Unable to get location: ' + (err && err.message ? err.message : String(err)));
  } finally { flyMeBtn.disabled = false; }
}

function setTrueNorth(){
  if(!viewer || !viewer.camera) return;
  const cart = Cesium.Ellipsoid.WGS84.cartesianToCartographic(viewer.camera.position);
  const lon = Cesium.Math.toDegrees(cart.longitude);
  const lat = Cesium.Math.toDegrees(cart.latitude);
  const height = cart.height || 2000;
  const currPitch = viewer.camera.pitch || Cesium.Math.toRadians(-45);
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
    orientation: { heading: 0.0, pitch: currPitch, roll: 0.0 },
    duration: 0.6
  });
}
if(flyMeBtn) flyMeBtn.addEventListener('click', flyToMyLocation);
if(trueNorthBtn) trueNorthBtn.addEventListener('click', setTrueNorth);

/* AdvancedVoxelManager v2
   - supports 4 families + subtypes
   - thin halo (blue/green/orange)
   - adjustable global opacity
*/
class AdvancedVoxelManager {
  constructor(viewer) {
    this.viewer = (typeof viewer === 'function') ? viewer() : viewer;
    this.entityTimestamps = new Map();
    this.tooltip = document.getElementById('voxelTooltip');

    // use the HUD DOM elements you actually have
    this.cloudCountEl = document.getElementById('cloudCount') || { textContent: "0" };
    this.rainCountEl  = document.getElementById('rainCount')  || { textContent: "0" };

    this._rotationCache = new Map();
    this._globalOpacity = parseFloat((document.getElementById('cloudOpacity')||{value:0.55}).value) || 0.55;
    this._baseAlpha = 1.0; // multiplier for per-voxel alpha
    this.startCleanupInterval();
    this.setupHover();
  }

  _upsertEllipsoid(id, lon, lat, centerHeight, radii, color, outlineColor, show=true) {
    let ent = this.viewer.entities.getById(id);
    const position = Cesium.Cartesian3.fromDegrees(lon, lat, centerHeight);
    const material = color;
    if (!ent) {
      ent = this.viewer.entities.add({
        id: id,
        position: position,
        ellipsoid: {
          radii: new Cesium.Cartesian3(radii[0], radii[1], radii[2]),
          material: material,
          outline: false,
          slicePartitions: 16,
          stacksPartitions: 8,
          show: show
        }
      });
    } else {
      ent.position = position;
      if (ent.ellipsoid) {
        ent.ellipsoid.radii = new Cesium.Cartesian3(radii[0], radii[1], radii[2]);
        ent.ellipsoid.material = material;
        ent.ellipsoid.show = show;
      } else {
        ent.ellipsoid = {
          radii: new Cesium.Cartesian3(radii[0], radii[1], radii[2]),
          material: material,
          outline: false,
          show: show
        };
      }
    }
    ent.__voxel = ent.__voxel || {};
    return ent;
  }

  _upsertEllipse(id, lon, lat, height, semiMajor, semiMinor, rotationRad, color, show=true) {
    let ent = this.viewer.entities.getById(id);
    const center = Cesium.Cartesian3.fromDegrees(lon, lat, height);
    if (!ent) {
      ent = this.viewer.entities.add({
        id: id,
        position: center,
        ellipse: {
          semiMajorAxis: semiMajor,
          semiMinorAxis: semiMinor,
          rotation: rotationRad || 0.0,
          extrudedHeight: 0.0,
          height: height,
          material: color,
          outline: false,
          show: show
        }
      });
    } else {
      ent.position = center;
      if (ent.ellipse) {
        ent.ellipse.semiMajorAxis = semiMajor;
        ent.ellipse.semiMinorAxis = semiMinor;
        ent.ellipse.rotation = rotationRad || 0.0;
        ent.ellipse.height = height;
        ent.ellipse.material = color;
        ent.ellipse.show = show;
      } else {
        ent.ellipse = {
          semiMajorAxis: semiMajor,
          semiMinorAxis: semiMinor,
          rotation: rotationRad || 0.0,
          height: height,
          material: color,
          show: show
        };
      }
    }
    ent.__voxel = ent.__voxel || {};
    return ent;
  }

  _upsertCylinder(id, lon, lat, centerHeight, length, radius, color, show=true) {
    let ent = this.viewer.entities.getById(id);
    const position = Cesium.Cartesian3.fromDegrees(lon, lat, centerHeight);
    if (!ent) {
      ent = this.viewer.entities.add({
        id: id,
        position: position,
        cylinder: { length: length, topRadius: radius, bottomRadius: radius, material: color, outline: false, show: show }
      });
    } else {
      ent.position = position;
      if (ent.cylinder) {
        ent.cylinder.length = length;
        ent.cylinder.topRadius = radius;
        ent.cylinder.bottomRadius = radius;
        ent.cylinder.material = color;
        ent.cylinder.show = show;
      } else {
        ent.cylinder = { length: length, topRadius: radius, bottomRadius: radius, material: color, show: show };
      }
    }
    ent.__voxel = ent.__voxel || {};
    return ent;
  }

  _haloColorForType(family, frac) {
    let base;
    if (family === 'high') base = Cesium.Color.fromBytes(180,220,255);
    else if (family === 'middle') base = Cesium.Color.fromBytes(120,220,200);
    else if (family === 'low') base = Cesium.Color.fromBytes(200,240,200);
    else base = Cesium.Color.fromBytes(255,200,140);
    const alpha = Math.max(0.08, Math.min(0.35, 0.10 + 0.25 * frac * this._globalOpacity));
    return base.withAlpha(alpha);
  }

  addOrUpdateVoxel(v) {
    if (!v || !v.id) return;
    const lat = Number(v.lat || v.latitude || 0);
    const lon = Number(v.lon || v.longitude || 0);

    const layer = (v.layer || 'cloud').toLowerCase();
    const frac = Number(v.cloud_frac || 0.0);
    const mmhr = Number(v.precip_mm_per_hr || (v.precip_rate_raw ? v.precip_rate_raw * 3600.0 : 0.0));
    const wind_dir = Number(v.wind_dir_deg || 0.0);

    // derive family
    let family = 'low';
    if (layer === 'high' || /cirr/i.test(v.cloud_type || '')) family = 'high';
    else if (layer === 'medium' || /alto/i.test(v.cloud_type || '')) family = 'middle';
    else if (layer === 'precip' || /nimb/i.test(v.cloud_type||'')) family = 'low';
    else if (/cumul/i.test(v.cloud_type||'') || layer==='vertical' || /convect/i.test(v.cloud_type||'')) family = 'vertical';
    const cloud_type = (v.cloud_type || '').toLowerCase();

    const baseAlpha = Math.max(0.06, Math.min(0.98, (0.18 + frac * 0.6) * this._globalOpacity));
    const iceTint = Cesium.Color.fromBytes(220,240,255).withAlpha(baseAlpha);
    const waterTint = Cesium.Color.fromBytes(200,220,255).withAlpha(baseAlpha);
    const greyTint = Cesium.Color.fromBytes(210,210,220).withAlpha(baseAlpha);
    const convectTint = Cesium.Color.fromBytes(255,230,200).withAlpha(Math.min(0.9, baseAlpha + 0.1));

    let mainColor;
    if (family === 'high') mainColor = iceTint;
    else if (family === 'middle') mainColor = waterTint;
    else if (family === 'vertical') mainColor = convectTint;
    else mainColor = greyTint;

    const haloColor = this._haloColorForType(family, frac);

    const size_xy = (Array.isArray(v.size_m) && v.size_m.length >= 2) ? Math.max(1200, v.size_m[0]) : Math.max(1200, 6000 * Math.max(0.2, frac));
    const thickness = Math.max(600, (v.thickness_m || 1600));
    const center_height = Number(v.height || (v.cloud_top ? (v.cloud_top - (thickness / 2)) : (v.cloud_top || 2000)));

    const alreadyExists = !!this.viewer.entities.getById(v.id);

    // precipitation
    if (family === 'precip' || v.layer === 'precip') {
      if (mmhr <= 0) return;
      const norm = Math.min(1.0, mmhr / Math.max(1e-6, RAIN_CLAMP || 10.0));
      const radius = Math.max(120, 200 + 4800.0 * norm);
      const length = Math.max(800, 3500 + 14000 * norm);
      const center = Math.max(length / 2.0 + 50.0, center_height);
      const rainColor = Cesium.Color.CYAN.withAlpha(Math.min(0.95, 0.6 + 0.4 * norm));
      const ent = this._upsertCylinder(v.id, lon, lat, center, length, radius, rainColor);
      ent.__voxel = v;
      this.entityTimestamps.set(v.id, Date.now());

      if (!alreadyExists) {
        const prevRain = parseInt(this.rainCountEl.textContent || "0");
        this.rainCountEl.textContent = String(prevRain + 1);
      }
      return;
    }

    // shape selection
    let ent;
    const isCirrus = /cirr/i.test(cloud_type) || family === 'high';
    const isStratusLike = /strat|nimb|stratus|strato/i.test(cloud_type) || (family === 'low' && frac > 0.03 && frac < 0.5);
    const isCumulus = /cumulonimbus|cumulus|convect/i.test(cloud_type) || family === 'vertical' || (family==='low' && frac > 0.45);
    const isMid = family === 'middle' || /alto/i.test(cloud_type);

    if (isCirrus) {
      const major = Math.max(2200, size_xy * 1.6);
      const minor = Math.max(300, size_xy * 0.28);
      const rotationRad = ((wind_dir || 0) * Math.PI / 180.0);
      ent = this._upsertEllipse(v.id, lon, lat, center_height, major, minor, rotationRad, mainColor);
      const haloId = `${v.id}__halo`;
      const haloRadii = [Math.max(major * 0.6, minor * 0.6), Math.max(major * 0.6, minor * 0.6), Math.max(800, thickness * 0.35)];
      this._upsertEllipsoid(haloId, lon, lat, center_height, haloRadii, haloColor);
    } else if (isCumulus) {
      const rxy = Math.max(1400, size_xy * 0.6);
      const rz = Math.max(1100, thickness * 0.9);
      ent = this._upsertEllipsoid(v.id, lon, lat, center_height, [rxy, rxy, rz], mainColor);
      const haloId = `${v.id}__halo`;
      const haloRadii = [rxy * 1.12, rxy * 1.12, rz * 1.05];
      this._upsertEllipsoid(haloId, lon, lat, center_height, haloRadii, haloColor);
    } else if (isMid) {
      const rxy = Math.max(1600, size_xy * 0.52);
      const rz = Math.max(850, thickness * 0.6);
      ent = this._upsertEllipsoid(v.id, lon, lat, center_height, [rxy, rxy * 0.85, rz], mainColor);
      const haloId = `${v.id}__halo`;
      const haloRadii = [rxy * 1.08, (rxy * 0.85) * 1.08, rz * 1.03];
      this._upsertEllipsoid(haloId, lon, lat, center_height, haloRadii, haloColor);
    } else if (isStratusLike) {
      const semiMajor = Math.max(1800, size_xy * 0.9);
      const semiMinor = Math.max(800, size_xy * 0.4);
      ent = this._upsertEllipse(v.id, lon, lat, center_height, semiMajor, semiMinor, 0.0, mainColor);
      const haloId = `${v.id}__halo`;
      const haloRadii = [Math.max(1000, semiMajor * 0.9), Math.max(1000, semiMinor * 0.9), Math.max(700, thickness * 0.5)];
      this._upsertEllipsoid(haloId, lon, lat, center_height, haloRadii, haloColor);
    } else {
      const rxy = Math.max(1200, size_xy * 0.45);
      const rz = Math.max(800, thickness * 0.5);
      ent = this._upsertEllipsoid(v.id, lon, lat, center_height, [rxy, rxy * 0.9, rz], mainColor);
      const haloId = `${v.id}__halo`;
      const haloRadii = [rxy * 1.08, (rxy * 0.9) * 1.08, rz * 1.02];
      this._upsertEllipsoid(haloId, lon, lat, center_height, haloRadii, haloColor);
    }

    if (ent) {
      ent.__voxel = v;
      this.entityTimestamps.set(v.id, Date.now());
    }

    if (!alreadyExists) {
      const prevCloud = parseInt(this.cloudCountEl.textContent || "0");
      this.cloudCountEl.textContent = String(prevCloud + 1);
    }

    this._applyOpacityToEntity(v.id);
  }

  _applyOpacityToEntity(id) {
    try {
      const ent = this.viewer.entities.getById(id);
      if (ent && ent.ellipsoid) {
        const mat = ent.ellipsoid.material;
        if (mat && mat.color && typeof mat.color.getValue === 'function') {
          const cval = mat.color.getValue(Cesium.JulianDate.now());
          if (cval) {
            ent.ellipsoid.material = cval.withAlpha((cval.alpha || 1.0) * this._globalOpacity);
          }
        } else if (ent.ellipsoid.material instanceof Cesium.Color) {
          ent.ellipsoid.material = ent.ellipsoid.material.withAlpha(ent.ellipsoid.material.alpha * this._globalOpacity);
        }
      }
      const halo = this.viewer.entities.getById(`${id}__halo`);
      if (halo && halo.ellipsoid) {
        const mat = halo.ellipsoid.material;
        if (mat && mat.color && typeof mat.color.getValue === 'function') {
          const cval = mat.color.getValue(Cesium.JulianDate.now());
          if (cval) halo.ellipsoid.material = cval.withAlpha(cval.alpha * this._globalOpacity);
        } else if (halo.ellipsoid.material instanceof Cesium.Color) {
          halo.ellipsoid.material = halo.ellipsoid.material.withAlpha(halo.ellipsoid.material.alpha * this._globalOpacity);
        }
      }
    } catch (e) { /* non-fatal */ }
  }

  setGlobalOpacity(opacity) {
    this._globalOpacity = Math.max(0, Math.min(1, Number(opacity)));
    this.viewer.entities.values.forEach(ent => {
      if (!ent || !ent.id || typeof ent.id !== 'string') return;
      if (!ent.__voxel) return;
      if (ent.__voxel.layer && ent.__voxel.layer.toLowerCase() === 'precip') return;
      try { this._applyOpacityToEntity(ent.id); } catch (e) {}
    });
  }

  startCleanupInterval() {
    setInterval(() => {
      const now = Date.now();
      if (!this.viewer || !this.viewer.entities) return;
      this.viewer.entities.values.forEach(ent => {
        if (!ent || !ent.id || typeof ent.id !== 'string') return;
        if (!ent.__voxel) return;
        const ts = this.entityTimestamps.get(ent.id) || 0;
        if (now - ts > 1000 * 60 * 45) {
          try {
            try { this.viewer.entities.removeById(`${ent.id}__halo`); } catch (_) {}
            this.viewer.entities.remove(ent);
          } catch (_) {}
          this.entityTimestamps.delete(ent.id);
        }
      });
    }, 60 * 1000);
  }

  setupHover() {
    const attempt = () => {
      if (!this.viewer || !this.viewer.scene || !this.viewer.scene.canvas) {
        setTimeout(attempt, 300);
        return;
      }
      const handler = new Cesium.ScreenSpaceEventHandler(this.viewer.scene.canvas);
      handler.setInputAction(movement => {
        const picked = this.viewer.scene.pick(movement.endPosition);
        if (picked && picked.id && picked.id.__voxel) {
          const v = picked.id.__voxel;
          let html = `<div style="font-weight:bold;margin-bottom:6px;color:#ffffff">${(v.cloud_type || v.layer || 'VOXEL').toUpperCase()}</div>`;
          html += `<div><strong>Lat, Lon:</strong> ${(v.lat||0).toFixed(4)}, ${(v.lon||0).toFixed(4)}</div>`;
          if (v.layer === 'precip') {
            html += `<div><strong>Center h:</strong> ${(v.height||0).toFixed(0)} m</div>`;
            html += `<div><strong>Rate:</strong> ${(v.precip_mm_per_hr||0).toFixed(2)} mm/hr</div>`;
          } else {
            html += `<div><strong>Top:</strong> ${(v.cloud_top||0).toFixed(0)} m</div>`;
            html += `<div><strong>Base:</strong> ${(v.cloud_base||0).toFixed(0)} m</div>`;
            html += `<div><strong>Cloud frac:</strong> ${(v.cloud_frac||0).toFixed(2)}</div>`;
          }
          if (v.wind_speed !== undefined) {
            html += `<div style="margin-top:6px;"><strong>Wind:</strong> ${v.wind_speed.toFixed(2)} m/s @ ${(v.wind_dir_deg||0).toFixed(0)}°</div>`;
          }
          html += `<div style="margin-top:6px;font-size:11px;color:#88ffff">voxel id: ${v.id}</div>`;
          this.tooltip.innerHTML = html;
          this.tooltip.style.display = 'block';
          this.tooltip.style.left = (movement.endPosition.x + 14) + 'px';
          this.tooltip.style.top  = (movement.endPosition.y + 14) + 'px';
        } else {
          this.tooltip.style.display = 'none';
        }
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    };
    attempt();
  }

  clearAll() {
    try {
      const ids = [];
      this.viewer.entities.values.forEach(ent => { if(ent && ent.id && typeof ent.id === 'string' && ent.id.startsWith('v_')) ids.push(ent.id); });
      ids.forEach(id => {
        try { this.viewer.entities.removeById(`${id}__halo`); } catch(_) {}
        try { this.viewer.entities.removeById(id); } catch(_) {}
        this.entityTimestamps.delete(id);
      });
      this.cloudCountEl.textContent = "0";
      this.rainCountEl.textContent = "0";
    } catch(e){ console.warn('clearAll failed', e); }
  }
}

// instantiate voxelManager and wire cloud opacity
const voxelManager = new AdvancedVoxelManager(() => window.viewer || viewer);
(function waitViewerReady(){
  const t = setInterval(()=> {
    if(window.viewer || viewer) {
      voxelManager.viewer = window.viewer || viewer;
      clearInterval(t);
    }
  }, 200);
})();

const cloudOpacityEl = document.getElementById('cloudOpacity');
const cloudOpacityVal = document.getElementById('cloudOpacityVal');
if (cloudOpacityEl) {
  cloudOpacityEl.addEventListener('input', () => {
    const val = parseFloat(cloudOpacityEl.value || 0.55);
    cloudOpacityVal.textContent = val.toFixed(2);
    try { voxelManager.setGlobalOpacity(val); } catch(e) { console.warn('setGlobalOpacity failed', e); }
  });
  try { voxelManager.setGlobalOpacity(parseFloat(cloudOpacityEl.value || 0.55)); } catch(e){}
}

/* SSE & Fetch Clouds */
let sseSource = null;
const fetchBtn = document.getElementById('fetchClouds');
let currentGeneration = null;

async function startSSE(){
  try{
    if(sseSource){ try{ sseSource.close(); }catch(e){} sseSource = null; fetchBtn.textContent='Fetch Clouds'; fetchBtn.classList.remove('btn-danger'); document.getElementById('statusLight').style.background='#f00'; return; }
    if(!viewer){ alert('Viewer not ready'); return; }
    const bounds = viewer.camera.computeViewRectangle(viewer.scene.globe.ellipsoid);
    if(!bounds){ alert('Cannot compute view extents. Zoom out or adjust view.'); return; }
    const north = Cesium.Math.toDegrees(bounds.north), south = Cesium.Math.toDegrees(bounds.south),
          west = Cesium.Math.toDegrees(bounds.west), east = Cesium.Math.toDegrees(bounds.east);
    const grid_n = Math.max(1, Math.min(18, parseInt(document.getElementById('gridN').value || 6)));
    const radios = document.getElementsByName('forecast'); let forecast='now'; for(const r of radios) if(r.checked){ forecast=r.value; break; }
    const url = `/voxels_stream?north=${north}&south=${south}&west=${west}&east=${east}&grid_n=${grid_n}&forecast=${encodeURIComponent(forecast)}`;
    console.info('Opening SSE:', url);
    fetchBtn.textContent='Stop Fetch'; fetchBtn.classList.add('btn-danger'); document.getElementById('statusLight').style.background='#0f0';
    currentGeneration = null;
    setCounts(0,0);
    // clear existing voxels before a new run
    try { voxelManager.clearAll(); } catch(e) {}

    sseSource = new EventSource(url);
    sseSource.onopen = ()=>{ console.info('SSE opened'); document.getElementById('statusLight').style.background='#0f0'; };
    sseSource.onerror = ev => {
      console.warn('SSE error', ev, sseSource ? sseSource.readyState : 'n/a');
      if(sseSource && sseSource.readyState === 2){ try{ sseSource.close(); }catch(e){} sseSource=null; fetchBtn.textContent='Fetch Clouds'; fetchBtn.classList.remove('btn-danger'); document.getElementById('statusLight').style.background='#f00'; }
      else { document.getElementById('statusLight').style.background='#f00'; }
    };

    sseSource.onmessage = e => {
      try {
        const msg = JSON.parse(e.data);
        if(!msg) return;
        if(msg.type === 'heartbeat') return;
        if(msg.type === 'debug'){ console.debug('SSE debug:', msg); return; }

        if(msg.type === 'init_gen'){
          const gen = msg.generation_id || msg.generation || null;
          if(currentGeneration && gen && currentGeneration !== gen){
            try { voxelManager.clearAll(); } catch(e){ console.warn('clearAll failed', e); }
            setCounts(0,0);
          }
          currentGeneration = gen;
          console.info('SSE init_gen, generation:', currentGeneration);
          return;
        }

        if(msg.type === 'voxels' && Array.isArray(msg.voxels)){
          const gen = msg.generation_id || currentGeneration;
          (msg.voxels || []).forEach(v => {
            try {
              if(v.precip_rate_raw !== undefined && v.precip_mm_per_hr === undefined) v.precip_mm_per_hr = v.precip_rate_raw * 3600.0;
              v.generation_id = gen;
              voxelManager.addOrUpdateVoxel(v);
            } catch(err){
              console.warn('addOrUpdateVoxel failed for voxel', v && v.id, err);
            }
          });
          return;
        }

        if(msg.type === 'mapped' || msg.type === 'processed'){ return; }
        if(msg.type === 'error'){ console.error('SSE payload error', msg); return; }

        // fallback: message contains voxels in top-level
        if(Array.isArray(msg.voxels)){
          msg.voxels.forEach(v => { try{ voxelManager.addOrUpdateVoxel(v); } catch(e){console.warn(e);} });
        }
      } catch(err){ console.warn('Failed to handle SSE message', err, e.data); }
    };

  } catch(err){
    console.error('startSSE failed', err);
    try{ if(sseSource){ sseSource.close(); sseSource=null; } }catch(e){}
    fetchBtn.textContent='Fetch Clouds'; fetchBtn.classList.remove('btn-danger'); document.getElementById('statusLight').style.background='#f00';
  }
}
fetchBtn.addEventListener('click', startSSE);
window.addEventListener('beforeunload', ()=> { if(sseSource) try{ sseSource.close(); }catch(e){} });

/* hide tooltip on mouseout */
(function attachCanvasMouseout(){
  const check = setInterval(()=>{ if(viewer && viewer.scene && viewer.scene.canvas){ viewer.scene.canvas.addEventListener('mouseout', ()=> { try{ document.getElementById('voxelTooltip').style.display = 'none'; } catch(e){} }); clearInterval(check);} }, 250);
})();
</script>
</body>
</html>


"""

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
        ee_ready=EE_READY,
        ee_msg=EE_INIT_MSG,
        ee_service_account=EE_SA,
        ee_project=EE_PROJECT
    )

# ----------------------------
# (retain the rest of your voxel fetch / SSE code below — omitted here for brevity)
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
