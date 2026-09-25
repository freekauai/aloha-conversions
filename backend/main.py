"""FastAPI app: upload an image, get a DST file and a stitch preview."""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import digitize as dz

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
    invert: bool = Form(False),
    spacing_mm: float = Form(1.3),
    color: str = Form("#ffffff"),
    triple_run: bool = Form(False),
    colors: int = Form(1),
    thread_colors: str | None = Form(None),  # JSON array of #rrggbb, one per detected block
    stitch_background: bool = Form(False),
):
    # Sync handler: FastAPI runs it in a worker thread, keeping the loop free.
    if mode not in ("fill", "outline"):
        raise HTTPException(422, "Stitch mode must be 'fill' or 'outline'.")
    if not 0 <= spacing_mm <= 3:
        raise HTTPException(422, "Letter spacing must be between 0 and 3 mm.")
    if not COLOR.match(color):
        raise HTTPException(422, "Thread color must look like #rrggbb.")
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
    opts = dz.Options(width_mm=width_mm, max_height_mm=max_height_mm, mode=mode,
                      invert=invert, spacing_mm=spacing_mm, triple_run=triple_run,
                      colors=colors, thread_colors=threads, stitch_background=stitch_background)

    # Files are written to a per-request temp dir (the DST must round-trip
    # through disk to be verified) and returned inline, so the engine keeps
    # no state between requests and can run as a serverless function.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = dz.digitize_to_files(image, opts, Path(tmp), color)
        except dz.DigitizeError as e:
            raise HTTPException(422, str(e))
        dst_b64 = base64.b64encode(result.dst_path.read_bytes()).decode()
        png_b64 = base64.b64encode(result.preview_path.read_bytes()).decode()

    v = result.verified
    return {
        "filename": safe_stem(file.filename),
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
        "dst_base64": dst_b64,
        "preview_png_base64": png_b64,
    }
