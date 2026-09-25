"""FastAPI app: upload an image, get a DST file and a stitch preview."""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import tempfile
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import digitize as dz
from . import threads as th

# Vercel caps request bodies at 4.5 MB; the pages shrink big images before upload.
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
MAX_IMAGE_SIDE = 4000
MIN_WIDTH_IN, MAX_WIDTH_IN = 0.5, 14.0
# Sites allowed to call the API from a browser (the kauaitoday.info page does).
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://kauaitoday.info,https://www.kauaitoday.info").split(",") if o.strip()]

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

# Per-IP limit on the expensive endpoints. The engine is stateless, so this
# lives in the memory of one function instance: it blunts a scripted client
# (which keeps hitting the same warm instance) rather than enforcing a hard
# global cap. Way above what a person clicking Generate can reach.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "30"))
RATE_WINDOW_SECONDS = int(os.environ.get("RATE_WINDOW_SECONDS", "600"))
RATE_LIMITED_PATHS = {"/api/digitize", "/api/colors"}
_hits: dict[str, deque[float]] = {}

# Usage stats go to the kauaitoday.info admin dashboard, server-side so they
# can't be spoofed or ad-blocked. Counts only: placement, fabric, colors,
# stitches. Off unless STATS_SECRET is set (it's shared with the site's
# api/submit.js). Best-effort: a failure never affects the conversion.
STATS_URL = os.environ.get("STATS_URL", "https://kauaitoday.info/api/submit")
STATS_SECRET = os.environ.get("STATS_SECRET", "")
STATS_TIMEOUT = 4


def report_stats(event: str, **fields) -> bool:
    if not STATS_SECRET:
        return False
    body = json.dumps({"kind": "aloha", "event": event, **fields}).encode()
    req = urllib.request.Request(STATS_URL, data=body, method="POST", headers={
        "Content-Type": "application/json", "x-aloha-secret": STATS_SECRET,
        "User-Agent": "AlohaConversions/1.0 (+https://kauaitoday.info/aloha-conversions)"})
    try:
        with urllib.request.urlopen(req, timeout=STATS_TIMEOUT) as r:
            return r.status == 200 and json.loads(r.read() or b"{}").get("ok") is True
    except Exception as e:  # network, 4xx/5xx, bot challenge: log and move on
        print(f"[stats] {event} not recorded: {e}")
        return False


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "?"


def rate_limited(ip: str, now: float | None = None) -> int:
    """Record a hit and return seconds until the next allowed request (0 = allowed)."""
    now = now or time.time()
    q = _hits.setdefault(ip, deque())
    while q and q[0] <= now - RATE_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return int(q[0] + RATE_WINDOW_SECONDS - now) + 1
    q.append(now)
    if len(_hits) > 10_000:  # keep a long-lived instance from growing forever
        for key in [k for k, v in _hits.items() if not v or v[-1] <= now - RATE_WINDOW_SECONDS]:
            _hits.pop(key, None)
    return 0

app = FastAPI(title="Image to DST")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], expose_headers=["Content-Disposition"])


@app.middleware("http")
async def guard(request: Request, call_next):
    # Cheap early rejection; the upload is re-checked after reading.
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
        return JSONResponse({"detail": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}, status_code=413)
    if request.method == "POST" and request.url.path in RATE_LIMITED_PATHS:
        wait = rate_limited(client_ip(request))
        if wait:
            minutes = max(1, round(wait / 60))
            return JSONResponse(
                {"detail": f"That's a lot of conversions. Take a break and try again in about {minutes} min."},
                status_code=429, headers={"Retry-After": str(wait)})
    return await call_next(request)


def image_dimensions(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG or JPEG header, without decoding pixels."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF:  # fill byte
                i += 1
            elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            elif marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # no length field
                i += 2
            else:
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        raise HTTPException(400, "Could not read the JPEG's dimensions.")
    raise HTTPException(415, "Only PNG and JPG images are supported.")


def decode_upload(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    w, h = image_dimensions(data)
    if w > MAX_IMAGE_SIDE or h > MAX_IMAGE_SIDE:
        raise HTTPException(413, f"Image is {w}x{h} px; the limit is {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE}.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(400, "The image could not be decoded.")
    return image


def safe_stem(filename: str | None) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename or "design").stem).strip("-")
    return stem[:40] or "design"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/presets")
def presets():
    return {
        "presets": [
            {"id": k, "label": p.label, "width_in": p.width_in, "max_height_in": p.max_height_in, "hint": p.hint}
            for k, p in dz.PRESETS.items()
        ],
        "custom": {"min_width_in": MIN_WIDTH_IN, "max_width_in": MAX_WIDTH_IN},
        "max_stitches": dz.MAX_STITCHES,
        "max_colors": dz.MAX_COLORS,
    }


@app.get("/api/threads")
def threads():
    return {"charts": th.CHARTS}


@app.get("/api/sewing")
def sewing():
    return {
        "fabrics": [{"id": k, "label": f.label, "row_spacing_mm": f.row_spacing_mm, "underlay": f.underlay,
                     "pull_comp_mm": f.pull_comp_mm, "hint": f.hint} for k, f in dz.FABRICS.items()],
        "densities": dz.DENSITIES,
        "underlay_types": list(dz.UNDERLAY_TYPES),
        "max_pull_comp_mm": dz.MAX_PULL_COMP_MM,
        "formats": [{"id": k, "label": v[0]} for k, v in dz.FORMATS.items()],
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/colors")
def colors_of(file: UploadFile = File(...), colors: int = Form(3)):
    """Detected color blocks for the multi-color flow, largest first. The
    digitize call reproduces the same blocks for the same image and count."""
    if not 2 <= colors <= dz.MAX_COLORS:
        raise HTTPException(422, f"Colors must be 2–{dz.MAX_COLORS}.")
    image = decode_upload(file.file.read(MAX_UPLOAD_BYTES + 1))
    try:
        blocks = dz.detect_colors(image, colors)
    except dz.DigitizeError as e:
        raise HTTPException(422, str(e))
    return {"blocks": [{"hex": b.hex, "area_pct": round(b.area_pct, 1), "is_background": b.is_background}
                       for b in blocks]}


@app.post("/api/digitize")
def digitize(
    file: UploadFile = File(...),
    preset: str = Form("hat"),
    width_in: float | None = Form(None),
    height_in: float | None = Form(None),
    mode: str = Form("fill"),
    invert: bool = Form(False),  # legacy; subject=dark
    subject: str = Form("auto"),  # auto | dark | light: which tone is the artwork
    spacing_mm: float = Form(1.3),
    color: str = Form("#ffffff"),
    triple_run: bool = Form(False),
    colors: int = Form(1),
    thread_colors: str | None = Form(None),  # JSON array of #rrggbb, one per detected block
    stitch_background: bool = Form(False),
    fabric: str | None = Form(None),  # a FABRICS id; sets the three below unless overridden
    density: str | None = Form(None),  # light | normal | dense
    row_spacing_mm: float | None = Form(None),
    angle_deg: float = Form(0),
    underlay: str | None = Form(None),
    pull_comp_mm: float | None = Form(None),
    formats: str = Form("dst"),  # comma-separated: dst,pes,jef,exp,vp3
    satin_max_mm: float = Form(6.0),  # strokes narrower than this get satin; 0 = tatami everything
):
    # Sync handler: FastAPI runs it in a worker thread, keeping the loop free.
    if mode not in ("fill", "outline"):
        raise HTTPException(422, "Stitch mode must be 'fill' or 'outline'.")
    if not 0 <= spacing_mm <= 3:
        raise HTTPException(422, "Letter spacing must be between 0 and 3 mm.")
    if not COLOR.match(color):
        raise HTTPException(422, "Thread color must look like #rrggbb.")
    if subject not in dz.SUBJECTS:
        raise HTTPException(422, "Subject must be auto, dark or light.")
    if not 1 <= colors <= dz.MAX_COLORS:
        raise HTTPException(422, f"Colors must be 1–{dz.MAX_COLORS}.")
    threads = None
    if thread_colors:
        try:
            threads = tuple(json.loads(thread_colors))
        except ValueError:
            raise HTTPException(422, "thread_colors must be a JSON array.")
        if not all(isinstance(t, str) and COLOR.match(t) for t in threads):
            raise HTTPException(422, "Every thread color must look like #rrggbb.")
    if preset == "custom" and not (width_in and MIN_WIDTH_IN <= width_in <= MAX_WIDTH_IN):
        raise HTTPException(422, f"Custom width must be {MIN_WIDTH_IN}–{MAX_WIDTH_IN} inches.")
    if preset == "custom" and height_in is not None and not (MIN_WIDTH_IN <= height_in <= MAX_WIDTH_IN):
        raise HTTPException(422, f"Custom height must be {MIN_WIDTH_IN}–{MAX_WIDTH_IN} inches.")

    image = decode_upload(file.file.read(MAX_UPLOAD_BYTES + 1))
    try:
        width_mm, max_height_mm = dz.resolve_size(preset, width_in, height_in)
    except dz.DigitizeError as e:
        raise HTTPException(422, str(e))
    # Sewing controls: fabric preset first, then any explicit overrides.
    spacing, under, pull = 0.42, "contour", 0.0
    if fabric:
        if fabric not in dz.FABRICS:
            raise HTTPException(422, f"Unknown fabric {fabric!r}.")
        f = dz.FABRICS[fabric]
        spacing, under, pull = f.row_spacing_mm, f.underlay, f.pull_comp_mm
    if density:
        if density not in dz.DENSITIES:
            raise HTTPException(422, "Density must be light, normal or dense.")
        spacing = dz.DENSITIES[density]
    if row_spacing_mm is not None:
        if not 0.3 <= row_spacing_mm <= 0.8:
            raise HTTPException(422, "Row spacing must be 0.3–0.8 mm.")
        spacing = row_spacing_mm
    if underlay is not None:
        if underlay not in dz.UNDERLAY_TYPES:
            raise HTTPException(422, "Underlay must be none, contour or full.")
        under = underlay
    if pull_comp_mm is not None:
        if not 0 <= pull_comp_mm <= dz.MAX_PULL_COMP_MM:
            raise HTTPException(422, f"Pull compensation must be 0–{dz.MAX_PULL_COMP_MM} mm.")
        pull = pull_comp_mm
    if not 0 <= satin_max_mm <= 12:
        raise HTTPException(422, "Satin width must be 0–12 mm.")
    if not -90 <= angle_deg <= 90:
        raise HTTPException(422, "Fill angle must be between -90 and 90 degrees.")
    wanted = tuple(dict.fromkeys(f.strip().lower() for f in formats.split(",") if f.strip()))
    bad = [f for f in wanted if f not in dz.FORMATS]
    if bad:
        raise HTTPException(422, f"Unknown format(s): {', '.join(bad)}.")

    opts = dz.Options(width_mm=width_mm, max_height_mm=max_height_mm, mode=mode,
                      invert=invert, subject=subject, spacing_mm=spacing_mm, triple_run=triple_run,
                      colors=colors, thread_colors=threads, stitch_background=stitch_background,
                      row_spacing_mm=spacing, angle_deg=angle_deg, underlay=under, pull_comp_mm=pull,
                      satin_max_mm=satin_max_mm)

    # Files are written to a per-request temp dir (the DST must round-trip
    # through disk to be verified) and returned inline, so the engine keeps
    # no state between requests and can run as a serverless function.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = dz.digitize_to_files(image, opts, Path(tmp), color, wanted or ("dst",),
                                          name=safe_stem(file.filename))
        except dz.DigitizeError as e:
            report_stats("error", preset=preset, fabric=fabric or "")
            raise HTTPException(422, str(e))
        dst_b64 = base64.b64encode(result.dst_path.read_bytes()).decode()
        png_b64 = base64.b64encode(result.preview_path.read_bytes()).decode()
        files_b64 = {fmt: base64.b64encode(path.read_bytes()).decode() for fmt, path in result.files.items()}

    v = result.verified
    recorded = report_stats("convert", preset=preset, fabric=fabric or "", colors=len(result.design.blocks),
                            stitches=v.stitch_count, formats=list(wanted or ("dst",)))
    return {
        "filename": safe_stem(file.filename),
        "stats_recorded": recorded,
        "subject": result.design.subject,  # single-color: which tone got stitched
        "stats": {
            "width_mm": round(v.width_mm, 1),
            "height_mm": round(v.height_mm, 1),
            "width_in": round(v.width_mm / dz.MM_PER_INCH, 2),
            "height_in": round(v.height_mm / dz.MM_PER_INCH, 2),
            "stitches": v.stitch_count,
            "trims": v.trim_count,
            "color_changes": v.color_changes,
            "sew_minutes": round(v.sew_minutes, 1),
            "shapes": len(result.design.shapes),
            "min_feature_mm": round(result.design.min_feature_mm, 1) if result.design.min_feature_mm else None,
        },
        "blocks": [
            {"detected": b.detected_hex, "thread": b.thread_hex, "area_pct": round(b.area_pct, 1),
             "shapes": len(b.shapes), "stitches": sum(len(r) + 4 for r in b.runs)}
            for b in result.design.blocks
        ],
        "warnings": result.design.warnings,
        "sewing": {"row_spacing_mm": spacing, "angle_deg": angle_deg, "underlay": under, "pull_comp_mm": pull,
                   "fabric": fabric, "satin_max_mm": satin_max_mm},
        "threads": {chart: [th.nearest(chart, b.thread_hex) for b in result.design.blocks] for chart in th.CHARTS},
        "dst_base64": dst_b64,
        "preview_png_base64": png_b64,
        "files_base64": files_b64,
    }
