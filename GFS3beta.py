#!/usr/bin/env python3
"""
FishVid — Three.js + 3d-tiles-renderer (local clone) + traffic proxy
Run: python3 server.py
Requirements: pip install quart hypercorn aiohttp
Place your cloned repo at: fishvid/3dTilesRendererJS
"""

import os
import asyncio
import logging
from urllib.parse import urlencode

from quart import Quart, Response, request, render_template_string
from hypercorn.asyncio import serve
from hypercorn.config import Config
import aiohttp

# ---------------- Config ----------------
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyANdFaa8vR69qKceJKvQBC9lyG6qb4ZPlQ")
PORT = int(os.environ.get("PORT", 8092))
SSL_CERT = os.environ.get("SSL_CERT", "/home/jayson_tolleson/security/fullchain.pem")
SSL_KEY = os.environ.get("SSL_KEY", "/home/jayson_tolleson/security/privkey.pem")
STATIC_FOLDER = "scripts"
# ---------------- App ----------------
app = Quart(__name__, static_folder=STATIC_FOLDER, static_url_path=f"/{STATIC_FOLDER}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fishvid-server")

# ---------------- HTML ----------------
INDEX_HTML = r"""
<!doctype html>
<html>
<style>
html,
map {
  height: 100%;
}
body {
  height: 100%;
  margin: 0;
  padding: 0;
}

</style>

  <head>
    <title>Map</title>
    <script type="module">
"use strict";
async function init() {
    // Import the needed libraries.
    const { Map3DElement, Model3DElement } = await google.maps.importLibrary("maps3d");
    const map = new Map3DElement({
        center: { lat: 34, lng: -118, altitude: 4395.4952 }, range: 1500, tilt: 74, heading: 0,
        mode: "HYBRID",
    });
    const model = new Model3DElement({
        src: 'https://maps-docs-team.web.app/assets/windmill.glb',
        position: { lat: 39.1178, lng: -106.4452, altitude: 4495.4952 },
        orientation: { heading: 0, tilt: 270, roll: 90 },
        scale: .15,
        altitudeMode: "CLAMP_TO_GROUND",
    });
    document.body.append(map);
    map.append(model);
}
init();
</script>
  </head>
  <body>
    <div id="map"></div>
    <!-- prettier-ignore -->
    <script>(g=>{var h,a,k,p="The Google Maps JavaScript API",c="google",l="importLibrary",q="__ib__",m=document,b=window;b=b[c]||(b[c]={});var d=b.maps||(b.maps={}),r=new Set,e=new URLSearchParams,u=()=>h||(h=new Promise(async(f,n)=>{await (a=m.createElement("script"));e.set("libraries",[...r]+"");for(k in g)e.set(k.replace(/[A-Z]/g,t=>"_"+t[0].toLowerCase()),g[k]);e.set("callback",c+".maps."+q);a.src=`https://maps.${c}apis.com/maps/api/js?`+e;d[q]=f;a.onerror=()=>h=n(Error(p+" could not load."));a.nonce=m.querySelector("script[nonce]")?.nonce||"";m.head.append(a)}));d[l]?console.warn(p+" only loads once. Ignoring:",g):d[l]=(f,...n)=>r.add(f)&&u().then(()=>d[l](f,...n))})
        ({key: "AIzaSyANdFaa8vR69qKceJKvQBC9lyG6qb4ZPlQ", v: "beta",});</script>
  </body>
</html>
"""

# ---------------- Routes ----------------

@app.route("/")
async def index():
    html = await render_template_string(
        INDEX_HTML,
        google_key=GOOGLE_KEY,
        static_folder=STATIC_FOLDER,
    )
    return Response(html, mimetype="text/html")



# ---------------- Run ----------------
if __name__ == "__main__":
    cfg = Config()
    cfg.bind = [f"0.0.0.0:{PORT}"]

    if os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY):
        cfg.certfile = SSL_CERT
        cfg.keyfile = SSL_KEY
        logger.info("✅ Running with TLS")
    else:
        logger.warning(f"⚠️ TLS cert/key not found; serving HTTP on 0.0.0.0:{PORT}")

    try:
        asyncio.run(serve(app, cfg))
    except KeyboardInterrupt:
        logger.info("shutting down")
