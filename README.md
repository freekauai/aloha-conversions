# Image to DST

A small web app that turns a PNG/JPG (logo or lettering) into a Tajima `.dst`
embroidery file, with a rendered stitch preview and stats before you download.

Single thread color, fill or outline stitching. No accounts, no database.

## Run locally

Needs Python 3.11+.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --reload
```

Open http://localhost:8000.

## Deploy to Vercel

Vercel detects FastAPI from the root `index.py` (which just imports `backend.main:app`)
and routes every request to it. From this directory:

```bash
vercel --prod
```

The first run creates the project. Production is public at
`https://<project>.vercel.app`; set `ALLOWED_ORIGINS` in the project's environment
variables if the calling site isn't kauaitoday.info.

## Run with Docker

```bash
docker build -t image-to-dst .
docker run --rm -p 8000:8000 image-to-dst
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## How it works

`backend/digitize.py` is the whole pipeline, as plain functions:

1. **Mask**: grayscale, threshold at 128 (light parts stitch, or dark with *Invert*),
   transparent pixels are background, 3×3 morphological open removes specks.
2. **Contours**: `cv2.findContours(RETR_CCOMP)` so letter counters (A, B, O, P, R)
   become polygon holes. Contours under 50 px² are dropped.
3. **Size**: scale to the target width, clamp to the preset's max height, center on (0, 0).
4. **Letter spacing**: glyphs (shapes grouped by overlapping x-range, so the dot of
   an *i* travels with its stem) are spread by `(i - (n-1)/2) × spacing`, then the
   design is refit to the target width.
5. **Inset** every shape 0.2 mm (mitre joins) so gaps and counters stay open after
   thread spread. Pieces under 0.5 mm² are dropped. The fit in step 3 adds 2 × 0.2 mm
   so the *stitched* design comes out exactly at the requested size.
6. **Simplify** at 0.08 mm.
7. **Order** shapes center-out (`abs(centroid.x)`), which is standard for hats.
8. **Stitch** each shape:
   - *Underlay*: running stitch (2.5 mm) around the shape inset 0.5 mm, holes included.
     Each lap continues along the ring (under where the fill will go) to the point
     nearest the next ring, so ring-to-ring hops rarely need a trim.
   - *Tatami fill*: horizontal rows 0.42 mm apart, stitches ≤ 3.0 mm, needle points
     staggered by `(row % 3) × 1.0 mm`, direction alternating each row. Rows chain
     into the next row when they overlap in x; otherwise a new run starts at the
     nearest top/bottom edge of an unfilled block.
   - *Outline only*: running stitch (2.0 mm) on the outline and holes, optional triple run.
9. **Travel rule**: a move becomes a stitch only if it is ≤ 2.5 mm *and* stays
   inside the shape (0.1 mm tolerance). Anything else is TRIM + JUMP. Every run gets
   short lock stitches at its start and end.
10. **Pattern**: TRIM after each shape, END at the finish, 0.1 mm DST units.
11. **Verify**: written with `pyembroidery.write_dst`, read back, and the reported stitch
    count, trim count and size come from the re-read file.

The preview is drawn from the re-read DST at 16 px/mm with 0.4 mm anti-aliased
lines in the thread color, breaking at trims/jumps, then downscaled 50%. The background
is dark, or light grey when the thread itself is very dark.

## Limits

- Uploads: PNG/JPG only, ≤ 10 MB, ≤ 4000 × 4000 px (checked from the file header before decoding).
- Output: ≤ 60,000 stitches, otherwise an error suggests a smaller size or Outline mode.
- Custom size: width 0.5–14 in, optional max height. Every named preset has a height cap
  (see `GET /api/presets`): hat front 4.25×2.25, hat side 2.5×1.5, hat back 4×1.5, beanie cuff 3.5×1.5,
  left chest 3.5×3.5, pocket 2.5×2.5, sleeve 3×2, youth front 7×7, full front 10×10, full back 12×12 in.
- Nothing is stored: the .dst and preview come back inline (base64) in the digitize
  response and the request's temp dir is deleted immediately. That's what lets the
  engine run as a serverless function.
- `MAX_UPLOAD_MB` (default 10) caps uploads; on Vercel it's 4 because the platform caps
  request bodies at 4.5 MB. Both pages shrink images over 3 MB to 2000 px before upload.
- Warns when the smallest feature (a letter, or a group of shapes stacked closely
  like *i* and its dot) is under 6 mm tall.
- Sew time is `stitches / 700` minutes; it doesn't count trims or color stops.

## Sewing controls

`GET /api/sewing` lists the fabric presets, densities, underlay types and file formats.
On `POST /api/digitize`:

- `fabric` (cap, tee, polo, fleece, towel, denim) sets density, underlay and pull
  compensation together; `density` (light 0.50 / normal 0.42 / dense 0.35 mm rows),
  `row_spacing_mm`, `underlay` (none / contour / full) and `pull_comp_mm` (0–0.5)
  override it individually.
- `angle_deg` (-90…90) rotates the fill rows. The fill is planned on the rotated shape
  and rotated back, so size and edges are unchanged.
- "full" underlay = contour plus a sparse 2 mm cross-hatch under the fill. Underlay
  travel inside a shape is walked with running stitches rather than trimmed, since
  the fill covers it.
- Pull compensation grows the shapes back out after the 0.2 mm inset (so the net
  inset is 0.2 − comp, and above 0.2 mm the design is slightly wider than requested,
  on purpose: the fabric pulls it back).
- `formats` (comma list of dst, pes, jef, exp, vp3) returns each file in
  `files_base64`. PES/JEF/VP3 carry the thread colors.

`GET /api/threads` returns real thread charts (Brother and Janome, from the tables
pyembroidery ships) and the digitize response lists the nearest chart thread per block.

## Rate limit

`POST /api/digitize` and `/api/colors` are limited per client IP to `RATE_LIMIT`
(default 30) requests per `RATE_WINDOW_SECONDS` (default 600); over that the engine
answers 429 with a `Retry-After`. The counter is in the memory of a function
instance, so on Vercel it blunts a scripted client (which keeps landing on the same
warm instance) rather than enforcing a hard global cap. Move it to Redis if that ever
matters.

## API

`GET /api/presets` lists the placements. `GET /api/health` for uptime checks.

`POST /api/digitize` (multipart): `file`, `preset` (an id from `/api/presets`, or `custom`),
`width_in` and optional `height_in` (custom only), `mode` (`fill` | `outline`), `invert`, `spacing_mm` (0–3),
`color` (`#rrggbb`), `triple_run`. Returns `stats`, `warnings`, `blocks`, a safe
`filename` stem, and the files inline as `dst_base64` and `preview_png_base64`.

## Using it from another site

The page at kauaitoday.info/aloha-conversions calls this API cross-origin. Allowed
origins come from `ALLOWED_ORIGINS` (comma-separated; default is
`https://kauaitoday.info,https://www.kauaitoday.info`). Set it on whichever host runs
this container, then put that host's URL in the page's `API_BASE`.

## Multi-color

Choose 2–6 colors and the engine k-means quantizes the image (in Lab space, so
distances match perception) into that many blocks. Transparent pixels belong to no
block. Without transparency, the color that owns most of the image border is flagged
as the page/background and skipped unless `stitch_background` is set.

- `POST /api/colors` (`file`, `colors`) returns the detected blocks, largest first,
  with a hex color, area % and `is_background`. Detection is deterministic, so the
  digitize call with the same image and count reproduces the same blocks.
- `POST /api/digitize` with `colors=N` and optional `thread_colors` (JSON array of
  `#rrggbb`, one per stitched block in that order) sews the blocks largest to smallest
  with a COLOR_CHANGE (a DST stop) between them. Stats include `color_changes` and the
  response lists each block's detected color, thread and stitch count.
- **Underlap:** each color is extended 0.8 mm under the colors sewn after it (about
  0.3 mm net once the seam gap and insets are taken out), so thread pull-in doesn't
  open a gap between neighbours. It never extends into the background.
- Sew time is still `stitches / 700`; each thread change adds operator time on top.
