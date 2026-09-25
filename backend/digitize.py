"""Image -> DST embroidery digitizing pipeline.

Every step is a plain function so it can be unit-tested on its own.
All geometry is in millimetres with +y pointing down, which matches both
image coordinates and pyembroidery's convention (the DST writer flips y).

Pipeline (see README for the rationale behind each constant):
  1. load_mask             threshold + morphological open
  2. extract_polygons      contours (RETR_CCOMP) -> shapely polygons with holes
  3. fit_to_size           scale to target width, clamp height, center on origin
  4. apply_letter_spacing  spread glyphs apart, then refit
  5. inset_shapes          pull edges in for thread spread
  6. simplify_shapes
  7. order_center_out
  8-9. plan_stitches       underlay + tatami fill (or outline), travel rule
  10. build_pattern        JUMP / STITCH / TRIM / END in 0.1 mm units
  11. write_and_verify     write DST, read back, count stitches and trims
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pyembroidery
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

Point = tuple[float, float]

MM_PER_INCH = 25.4

# 1-2: image cleanup
THRESHOLD = 128
MIN_CONTOUR_AREA_PX = 50

# 5-6: geometry cleanup
INSET_MM = 0.2
MIN_PIECE_AREA_MM2 = 0.5
SIMPLIFY_MM = 0.08

# 8: stitch generation
UNDERLAY_INSET_MM = 0.5
UNDERLAY_STITCH_MM = 2.5
FILL_ROW_SPACING_MM = 0.42
FILL_STITCH_MM = 3.0
FILL_MIN_STITCH_MM = 0.5
FILL_EDGE_MM = 0.1  # first/last fill row sits this far inside the shape's top/bottom  # stagger points closer than this to a row end are dropped
OUTLINE_STITCH_MM = 2.0
MIN_STITCH_MM = 0.1  # consecutive needle points closer than this are merged
TIE_MM = 0.5  # lock-stitch length at the start and end of every run

# 9: travel rule
MAX_TRAVEL_MM = 2.5
TRAVEL_TOLERANCE_MM = 0.1  # slack for "does the travel line stay inside the shape"

# Multi-color
MAX_COLORS = 6
MIN_BLOCK_PCT = 1.0  # clusters smaller than this (anti-aliased edges) merge into a neighbour
UNDERLAP_MM = 0.8  # a color reaches this far under the next one (~0.3 mm net after the seam gap and both insets)
SEAM_GAP_MM = 0.15  # pixel masks leave ~0.1 mm between neighbouring colors; bridge it

# Limits and reporting
MAX_STITCHES = 60_000  # a 10 in fill runs ~50k; 30k made the large presets unusable
SMALL_FEATURE_MM = 6.0
STITCHES_PER_MINUTE = 700
DST_UNITS_PER_MM = 10  # DST / pyembroidery units are 0.1 mm

@dataclass(frozen=True)
class Preset:
    label: str
    width_in: float
    max_height_in: float | None
    hint: str


# Standard garment placement fields. Width is the target width; height is the
# largest the field allows, so a tall design is scaled down to fit it.
PRESETS: dict[str, Preset] = {
    "hat": Preset("Hat front", 4.25, 2.25, "Structured cap front panel"),
    "hat_side": Preset("Hat side", 2.5, 1.5, "Small mark on the side of a cap"),
    "hat_back": Preset("Hat back", 4.0, 1.5, "Arc above the closure"),
    "beanie": Preset("Beanie cuff", 3.5, 1.5, "Folded cuff of a knit beanie"),
    "left_chest": Preset("Left chest", 3.5, 3.5, "Polo, tee and jacket logo"),
    "pocket": Preset("Pocket", 2.5, 2.5, "Shirt pocket or above-pocket mark"),
    "sleeve": Preset("Sleeve", 3.0, 2.0, "Upper sleeve of a tee or jacket"),
    "youth_front": Preset("Youth front", 7.0, 7.0, "Center chest on youth sizes"),
    "full_front": Preset("Full front", 10.0, 10.0, "Center chest on tees and hoodies"),
    "full_back": Preset("Full back", 12.0, 12.0, "Jacket and hoodie back"),
}


@dataclass(frozen=True)
class Fabric:
    label: str
    row_spacing_mm: float
    underlay: str
    pull_comp_mm: float
    hint: str


# What a digitizer would set for each garment. Stretchy or plush fabrics need
# more underlay (to hold the loft down) and more pull compensation (the fabric
# gives, so shapes shrink across the stitch direction).
FABRICS: dict[str, Fabric] = {
    "cap": Fabric("Structured cap", 0.40, "contour", 0.15, "Buckram-backed front panel: stable, little pull"),
    "tee": Fabric("Cotton tee", 0.42, "contour", 0.15, "Light jersey; keep density moderate to avoid puckering"),
    "polo": Fabric("Polo pique", 0.42, "full", 0.25, "Textured knit; full underlay keeps the fill from sinking"),
    "fleece": Fabric("Fleece / hoodie", 0.40, "full", 0.30, "Plush and stretchy; full underlay and extra compensation"),
    "towel": Fabric("Towel", 0.38, "full", 0.35, "Loops swallow stitches; dense fill plus full underlay"),
    "denim": Fabric("Denim / canvas", 0.42, "contour", 0.10, "Stable heavy fabric; least compensation"),
}
DENSITIES = {"light": 0.50, "normal": 0.42, "dense": 0.35}
UNDERLAY_TYPES = ("none", "contour", "full")
UNDERLAY_FILL_SPACING_MM = 2.0  # "full" underlay: sparse cross-hatch rows
MAX_PULL_COMP_MM = 0.5


class DigitizeError(ValueError):
    """A user-facing problem with the input or the requested size."""


@dataclass(frozen=True)
class Options:
    width_mm: float
    max_height_mm: float | None = None
    mode: str = "fill"  # "fill" | "outline"
    invert: bool = False
    spacing_mm: float = 1.3
    triple_run: bool = False
    colors: int = 1  # 1 = threshold (single thread); 2..MAX_COLORS = k-means blocks
    thread_colors: tuple[str, ...] | None = None  # hex per block, sewing order; None = detected
    stitch_background: bool = False  # multi-color: also sew the page/background color
    # Sewing controls (see FABRICS for the presets that set them together)
    row_spacing_mm: float = 0.42  # tatami density; smaller = denser
    angle_deg: float = 0.0  # fill direction, counter-clockwise from horizontal
    underlay: str = "contour"  # "none" | "contour" | "full" (contour + sparse cross fill)
    pull_comp_mm: float = 0.0  # grow shapes back out by this much to counter thread pull-in


@dataclass
class Design:
    shapes: list[Polygon]  # final stitched shapes, in stitch order
    runs: list[list[Point]]  # continuous stitch runs; TRIM + JUMP between runs
    min_feature_mm: float | None
    warnings: list[str] = field(default_factory=list)
    blocks: list["BlockPlan"] = field(default_factory=list)  # per color, sewing order


@dataclass
class Verified:
    pattern: pyembroidery.EmbPattern  # as re-read from the DST file
    stitch_count: int
    trim_count: int
    color_changes: int
    bounds_mm: tuple[float, float, float, float]  # minx, miny, maxx, maxy of stitches

    @property
    def width_mm(self) -> float:
        return self.bounds_mm[2] - self.bounds_mm[0]

    @property
    def height_mm(self) -> float:
        return self.bounds_mm[3] - self.bounds_mm[1]

    @property
    def sew_minutes(self) -> float:
        """Run time only; thread changes add operator time on top."""
        return self.stitch_count / STITCHES_PER_MINUTE


def resolve_size(preset: str, custom_width_in: float | None = None,
                 custom_height_in: float | None = None) -> tuple[float, float | None]:
    """Return (width_mm, max_height_mm) for a preset name or a custom size."""
    if preset == "custom":
        if custom_width_in is None or custom_width_in <= 0:
            raise DigitizeError("Custom preset needs a width in inches.")
        max_h = custom_height_in * MM_PER_INCH if custom_height_in and custom_height_in > 0 else None
        return custom_width_in * MM_PER_INCH, max_h
    if preset not in PRESETS:
        raise DigitizeError(f"Unknown preset {preset!r}.")
    p = PRESETS[preset]
    return p.width_in * MM_PER_INCH, p.max_height_in * MM_PER_INCH if p.max_height_in else None


# --------------------------------------------------------------------------- 1


def load_mask(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Binary uint8 mask (255 = stitch) from a gray, BGR or BGRA image.

    Light pixels are stitched by default; `invert` stitches the dark ones.
    Transparent pixels are always background.
    """
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:  # 16-bit PNGs
        gray = (gray / 257).astype(np.uint8)
        if alpha is not None:
            alpha = (alpha / 257).astype(np.uint8)

    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, THRESHOLD - 1, 255, mode)  # light = gray >= 128
    if alpha is not None:
        mask[alpha < THRESHOLD] = 0
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------------------------------- 2


def polygons_of(geom: BaseGeometry) -> list[Polygon]:
    """Flatten any geometry into its non-empty polygons."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in polygons_of(g)]
    return []


def extract_polygons(mask: np.ndarray) -> list[Polygon]:
    """Contours -> polygons in pixel space. Letter counters become holes."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hier = hierarchy[0]
    polys: list[Polygon] = []
    for i, contour in enumerate(contours):
        if hier[i][3] != -1:  # a hole; handled with its parent
            continue
        if len(contour) < 3 or cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
            continue
        holes = []
        child = hier[i][2]
        while child != -1:
            hole = contours[child]
            if len(hole) >= 3 and cv2.contourArea(hole) >= MIN_CONTOUR_AREA_PX:
                holes.append(hole.reshape(-1, 2).astype(float))
            child = hier[child][0]
        poly = Polygon(contour.reshape(-1, 2).astype(float), holes)
        polys.extend(polygons_of(shapely.make_valid(poly)))
    return polys


# --------------------------------------------------------------------------- 3


def union_bounds(polys: list[Polygon]) -> tuple[float, float, float, float]:
    b = np.array([p.bounds for p in polys])
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


def fit_to_size(polys: list[Polygon], width: float, max_height: float | None = None) -> list[Polygon]:
    """Uniformly scale so the design is `width` wide (or `max_height` tall if
    that is the tighter limit), centered on (0, 0)."""
    minx, miny, maxx, maxy = union_bounds(polys)
    w, h = maxx - minx, maxy - miny
    scale = width / w
    if max_height is not None and h * scale > max_height:
        scale = max_height / h
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return [
        affinity.scale(affinity.translate(p, -cx, -cy), scale, scale, origin=(0, 0))
        for p in polys
    ]


# --------------------------------------------------------------------------- 4


def glyph_clusters(polys: list[Polygon]) -> list[list[int]]:
    """Group shapes whose x-ranges overlap (the dot of an i, the pieces of a
    broken letter) so letter spacing moves them together. Left to right."""
    order = sorted(range(len(polys)), key=lambda i: polys[i].bounds[0])
    clusters: list[list[int]] = []
    right = -math.inf
    for i in order:
        minx, _, maxx, _ = polys[i].bounds
        if clusters and minx < right:
            clusters[-1].append(i)
            right = max(right, maxx)
        else:
            clusters.append([i])
            right = maxx
    return clusters


def apply_letter_spacing(polys: list[Polygon], spacing_mm: float) -> list[Polygon]:
    """Translate the i-th glyph (by min-x) by (i - (n-1)/2) * spacing."""
    clusters = glyph_clusters(polys)
    n = len(clusters)
    out = list(polys)
    for idx, members in enumerate(clusters):
        dx = (idx - (n - 1) / 2) * spacing_mm
        for i in members:
            out[i] = affinity.translate(out[i], dx, 0)
    return out


def feature_groups(polys: list[Polygon]) -> list[list[int]]:
    """Group shapes into features for size checks: shapes that overlap in x
    and are vertically close (gap no bigger than the smaller shape, like the
    dot of an i) join. Unlike glyph_clusters, separate lines of text stay apart."""
    parent = list(range(len(polys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bounds = [p.bounds for p in polys]
    for i, a in enumerate(bounds):
        for j in range(i + 1, len(bounds)):
            b = bounds[j]
            if a[0] < b[2] and b[0] < a[2]:
                gap = max(b[1] - a[3], a[1] - b[3])
                if gap <= min(a[3] - a[1], b[3] - b[1]):
                    parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(polys)):
        groups[find(i)].append(i)
    return list(groups.values())


def min_feature_height(polys: list[Polygon]) -> float | None:
    heights = []
    for members in feature_groups(polys):
        _, miny, _, maxy = union_bounds([polys[i] for i in members])
        heights.append(maxy - miny)
    return min(heights) if heights else None


# --------------------------------------------------------------------------- 5-7


def inset_shapes(polys: list[Polygon], inset_mm: float = INSET_MM) -> list[Polygon]:
    out = []
    for p in polys:
        for piece in polygons_of(p.buffer(-inset_mm, join_style="mitre")):
            if piece.area >= MIN_PIECE_AREA_MM2:
                out.append(piece)
    return out


def simplify_shapes(polys: list[Polygon], tolerance: float = SIMPLIFY_MM) -> list[Polygon]:
    out = []
    for p in polys:
        out.extend(polygons_of(p.simplify(tolerance, preserve_topology=True)))
    return out


def order_center_out(polys: list[Polygon]) -> list[Polygon]:
    return sorted(polys, key=lambda p: (abs(p.centroid.x), p.centroid.y))


# --------------------------------------------------------------------------- 8-9


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def subdivide(points: list[Point], max_len: float) -> list[Point]:
    """Split every edge into equal pieces no longer than max_len (keeps corners)."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(_dist(a, b) / max_len))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def rings_of(poly: Polygon) -> list[list[Point]]:
    """Exterior and hole rings as open coordinate lists."""
    rings = [poly.exterior] + list(poly.interiors)
    return [[(x, y) for x, y in r.coords[:-1]] for r in rings if len(r.coords) > 3]


def closed_from(ring: list[Point], near: Point) -> list[Point]:
    """Rotate a ring to start at the vertex nearest `near`, and close it."""
    i = min(range(len(ring)), key=lambda k: _dist(ring[k], near))
    rotated = ring[i:] + ring[:i]
    return rotated + [rotated[0]]


def _closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Indices of the closest vertices between point sets a and b, and their distance."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return int(i), int(j), float(d[i, j])


def _lap_with_tail(ring: list[Point], entry: int, exit_: int) -> list[Point]:
    """A full lap of an open ring starting at vertex `entry`, then onward to
    vertex `exit_`, in whichever direction makes that tail shorter."""
    n = len(ring)

    def arc(step: int) -> tuple[list[Point], float]:
        idx = [(entry + step * k) % n for k in range(n + 1)]
        k = 0
        while (entry + step * k) % n != exit_:
            k += 1
        tail = [(entry + step * m) % n for m in range(1, k + 1)]
        pts = [ring[i] for i in idx + tail]
        return pts, sum(_dist(ring[a], ring[b]) for a, b in zip([entry] + tail, tail))

    fwd, fwd_len = arc(1)
    back, back_len = arc(-1)
    return fwd if fwd_len <= back_len else back


def fill_segments(shape: Polygon, spacing: float = FILL_ROW_SPACING_MM) -> list[tuple[int, float, float, float]]:
    """Horizontal scanline spans (row index k, y, x0, x1). Rows are spaced
    evenly within the shape, no further apart than `spacing`, with the first
    and last rows FILL_EDGE_MM inside the top and bottom so the fill reaches
    the edge instead of stopping wherever a global grid happened to land."""
    minx, miny, maxx, maxy = shape.bounds
    top, bottom = miny + FILL_EDGE_MM, maxy - FILL_EDGE_MM
    if bottom <= top:
        ys = np.array([(miny + maxy) / 2])
    else:
        n = math.ceil((bottom - top) / spacing) + 1
        ys = np.linspace(top, bottom, n)
    lines = shapely.linestrings(
        np.stack(
            [np.full_like(ys, minx - 1), ys, np.full_like(ys, maxx + 1), ys], axis=1
        ).reshape(-1, 2, 2)
    )
    segs = []
    for k, (y, inter) in enumerate(zip(ys, shapely.intersection(shape, lines))):
        parts = getattr(inter, "geoms", [inter])
        spans = []
        for g in parts:
            if isinstance(g, LineString) and not g.is_empty:
                xs = [c[0] for c in g.coords]
                if max(xs) - min(xs) > 1e-6:
                    spans.append((min(xs), max(xs)))
        for x0, x1 in sorted(spans):
            segs.append((k, float(y), x0, x1))
    return segs


def row_points(k: int, y: float, x0: float, x1: float, forward: bool,
               length: float = FILL_STITCH_MM) -> list[Point]:
    """Needle points across one tatami row, staggered by (k % 3) * length / 3."""
    offset = (k % 3) * length / 3
    m0 = math.ceil((x0 + FILL_MIN_STITCH_MM - offset) / length)
    m1 = math.floor((x1 - FILL_MIN_STITCH_MM - offset) / length)
    xs = [x0] + [m * length + offset for m in range(m0, m1 + 1)] + [x1]
    pts = subdivide([(x, y) for x in xs], length)  # caps end gaps at `length`
    return pts if forward else pts[::-1]


def _overlaps(a: tuple[int, float, float, float], b: tuple[int, float, float, float]) -> bool:
    return a[2] < b[3] and b[2] < a[3]


class StitchPlanner:
    """Accumulates continuous stitch runs and enforces the travel rule."""

    def __init__(self) -> None:
        self.runs: list[list[Point]] = []
        self.run: list[Point] | None = None
        self.region = None
        # Underlay is covered by the fill, so a long move that stays inside the
        # shape can be walked with running stitches instead of trimmed.
        self.walk = False

    @property
    def pos(self) -> Point:
        if self.run:
            return self.run[-1]
        if self.runs:
            return self.runs[-1][-1]
        return (0.0, 0.0)

    def travel_to(self, p: Point) -> None:
        """Stitch to p only if the move is short and stays inside the current
        shape; otherwise end the run (TRIM) and start a new one (JUMP)."""
        if self.run:
            last = self.run[-1]
            d = _dist(last, p)
            inside = d < 1e-9 or self.region.covers(LineString([last, p]))
            if d <= MAX_TRAVEL_MM and inside:
                self.stitch(p)
                return
            if self.walk and inside:
                for q in subdivide([last, p], MAX_TRAVEL_MM)[1:]:
                    self.stitch(q)
                return
        self.break_run()
        self.run = [p]

    def stitch(self, p: Point) -> None:
        if self.run is None:
            self.run = [p]
        elif _dist(self.run[-1], p) >= MIN_STITCH_MM:
            self.run.append(p)

    def path(self, pts: list[Point]) -> None:
        self.travel_to(pts[0])
        for p in pts[1:]:
            self.stitch(p)

    def break_run(self) -> None:
        if self.run and len(self.run) >= 2:
            self.runs.append(self.run)
        self.run = None

    # -- per-shape strategies ------------------------------------------------

    def running_rings(self, rings: list[list[Point]], length: float, triple: bool = False) -> None:
        """Outline mode: running stitch around each ring, nearest ring first."""
        rings = list(rings)
        while rings:
            here = self.pos
            ring = min(rings, key=lambda r: min(_dist(v, here) for v in r))
            rings.remove(ring)
            self._ring(closed_from(ring, here), length, triple)

    def underlay_rings(self, rings: list[list[Point]], length: float, end_near: Point | None) -> None:
        """Underlay: after each full lap, keep stitching along the ring (hidden
        under the fill) to the vertex nearest the next ring, so the hop between
        rings is short and stays inside the shape instead of forcing a trim.
        The last ring finishes nearest `end_near` (where the fill starts)."""
        rings = [np.asarray(r) for r in rings]
        order = []
        ref = np.asarray([self.pos])
        while rings:
            j = min(range(len(rings)), key=lambda k: _closest_pair(ref, rings[k])[2])
            order.append(rings.pop(j))
            ref = order[-1]
        for n, ring in enumerate(order):
            entry = _closest_pair(np.asarray([self.pos]), ring)[1]
            if n + 1 < len(order):
                exit_ = _closest_pair(ring, order[n + 1])[0]
            elif end_near is not None:
                exit_ = _closest_pair(ring, np.asarray([end_near]))[0]
            else:
                exit_ = entry
            self.path(subdivide(_lap_with_tail([tuple(p) for p in ring], entry, exit_), length))

    def _ring(self, closed: list[Point], length: float, triple: bool) -> None:
        pts = subdivide(closed, length)
        if triple:
            tripled = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                tripled += [b, a, b]
            pts = tripled
        self.path(pts)

    def fill_start(self, segs: list[tuple[int, float, float, float]], unvisited: set[int],
                   by_row: dict[int, list[int]]) -> tuple[int, bool, int]:
        """Nearest endpoint among segments at the top or bottom edge of an
        unvisited block, so a run sweeps the whole block in one direction.
        Returns (segment index, left-to-right?, vertical direction)."""
        here = self.pos
        best = None
        for i in unvisited:
            k, y, x0, x1 = segs[i]
            above = any(j in unvisited and _overlaps(segs[i], segs[j]) for j in by_row.get(k - 1, ()))
            below = any(j in unvisited and _overlaps(segs[i], segs[j]) for j in by_row.get(k + 1, ()))
            if above and below:
                continue
            vdir = 1 if not above else -1
            for x, forward in ((x0, True), (x1, False)):
                d = (x - here[0]) ** 2 + (y - here[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, i, forward, vdir)
        assert best is not None  # the topmost unvisited row always qualifies
        return best[1], best[2], best[3]

    def tatami(self, shape: Polygon, segs: list[tuple[int, float, float, float]]) -> None:
        by_row: dict[int, list[int]] = defaultdict(list)
        for i, (k, _, _, _) in enumerate(segs):
            by_row[k].append(i)
        unvisited = set(range(len(segs)))
        while unvisited:
            i, forward, vdir = self.fill_start(segs, unvisited, by_row)
            while True:
                unvisited.discard(i)
                k, y, x0, x1 = segs[i]
                self.path(row_points(k, y, x0, x1, forward))
                nxt = [j for j in by_row.get(k + vdir, ())
                       if j in unvisited and _overlaps(segs[i], segs[j])]
                if not nxt:
                    break
                end_x = x1 if forward else x0
                forward = not forward
                i = min(nxt, key=lambda j: abs((segs[j][2] if forward else segs[j][3]) - end_x))


def _rotate(points: list[Point], deg: float) -> list[Point]:
    if not deg:
        return points
    c, si = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(x * c - y * si, x * si + y * c) for x, y in points]


def plan_stitches(shapes: list[Polygon], mode: str = "fill", triple_run: bool = False,
                  row_spacing: float = FILL_ROW_SPACING_MM, angle_deg: float = 0.0,
                  underlay: str = "contour") -> list[list[Point]]:
    """Stitch runs for the shapes, in order. The fill is planned on the shape
    rotated by -angle (so rows are horizontal), then rotated back."""
    planner = StitchPlanner()
    for shape in shapes:
        first_run = len(planner.runs)
        work = affinity.rotate(shape, -angle_deg, origin=(0, 0)) if angle_deg else shape
        planner.region = prep(work.buffer(TRAVEL_TOLERANCE_MM))
        if mode == "outline":
            planner.running_rings(rings_of(work), OUTLINE_STITCH_MM, triple=triple_run)
        else:
            segs = fill_segments(work, row_spacing)
            # Pick where the fill will begin, then end the underlay right there
            # so the hop from underlay to fill is a short stitch, not a trim.
            start = None
            if segs:
                by_row: dict[int, list[int]] = defaultdict(list)
                for i, (k, _, _, _) in enumerate(segs):
                    by_row[k].append(i)
                i, forward, _ = planner.fill_start(segs, set(range(len(segs))), by_row)
                _, y, x0, x1 = segs[i]
                start = (x0 if forward else x1, y)
            if underlay != "none":
                planner.walk = True
                inner = polygons_of(work.buffer(-UNDERLAY_INSET_MM, join_style="mitre"))
                if underlay == "full":
                    # Sparse rows across the fill direction, then the contour.
                    for piece in inner:
                        cross = affinity.rotate(piece, 90, origin=(0, 0))
                        planner.region = prep(cross.buffer(TRAVEL_TOLERANCE_MM))
                        n = len(planner.runs)
                        planner.tatami(cross, fill_segments(cross, UNDERLAY_FILL_SPACING_MM))
                        planner.break_run()
                        for r in range(n, len(planner.runs)):
                            planner.runs[r] = _rotate(planner.runs[r], -90)
                        if planner.run:
                            planner.run = _rotate(planner.run, -90)
                    planner.region = prep(work.buffer(TRAVEL_TOLERANCE_MM))
                planner.underlay_rings([r for p in inner for r in rings_of(p)], UNDERLAY_STITCH_MM, end_near=start)
                planner.walk = False
            planner.tatami(work, segs)
        planner.break_run()  # TRIM after each shape
        if angle_deg:
            for r in range(first_run, len(planner.runs)):
                planner.runs[r] = _rotate(planner.runs[r], angle_deg)
    return planner.runs


# --------------------------------------------------------------------------- colors


@dataclass
class ColorBlock:
    """One quantized color of the image. Blocks are ordered largest area first."""
    hex: str
    area_pct: float
    is_background: bool
    mask: np.ndarray  # uint8, 255 where this color is


def _pixels_and_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif image.shape[2] == 4:
        bgr, alpha = image[:, :, :3], image[:, :, 3]
    else:
        bgr, alpha = image, None
    if bgr.dtype != np.uint8:
        bgr = (bgr / 257).astype(np.uint8)
        alpha = (alpha / 257).astype(np.uint8) if alpha is not None else None
    return np.ascontiguousarray(bgr), alpha


def detect_colors(image: np.ndarray, n: int) -> list[ColorBlock]:
    """K-means the image into n colors (in Lab, so distances match perception).
    Deterministic for a given image and n, so a later call reproduces the same
    blocks. Clusters under MIN_BLOCK_PCT (anti-aliased edges) are merged into
    the nearest color. The color owning most of the image border is the page:
    the part of it connected to the border becomes a background block, while
    any islands of that color inside the art stay as their own block.
    Transparent pixels belong to no block."""
    n = max(2, min(MAX_COLORS, int(n)))
    bgr, alpha = _pixels_and_alpha(image)
    opaque = np.ones(bgr.shape[:2], bool) if alpha is None else alpha >= THRESHOLD
    total = int(opaque.sum())
    if total < n:
        raise DigitizeError("The image is almost entirely transparent.")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    idx = np.flatnonzero(opaque.ravel())
    sample = idx[:: max(1, len(idx) // 200_000)]  # even stride keeps it deterministic
    cv2.setRNGSeed(7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(lab[sample], n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    def assign(cs: np.ndarray) -> np.ndarray:
        d = ((lab[idx, None, :] - cs[None, :, :]) ** 2).sum(axis=2)
        labels = np.full(lab.shape[0], -1, np.int32)
        labels[idx] = d.argmin(axis=1)
        return labels.reshape(bgr.shape[:2])

    labels = assign(centers)
    counts = np.bincount(labels[labels >= 0], minlength=len(centers))
    keep = counts >= total * MIN_BLOCK_PCT / 100
    if keep.sum() >= 2 and not keep.all():
        centers = centers[keep]
        labels = assign(centers)

    def hex_of(center: np.ndarray) -> str:
        b, g, r = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2BGR)[0, 0]
        return "#%02x%02x%02x" % (r, g, b)

    def block(mask: np.ndarray, center: np.ndarray, is_bg: bool) -> ColorBlock | None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        area = int(np.count_nonzero(mask))
        if area == 0:
            return None
        return ColorBlock(hex_of(center), 100 * area / total, is_bg, mask)

    # The page: the color that owns most of the (opaque) border.
    h, w = labels.shape
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg_label = -1
    if len(border) and np.count_nonzero(border >= 0) >= len(border) * 0.5:
        bc = np.bincount(border[border >= 0], minlength=len(centers))
        if bc.max() > len(border) * 0.5:
            bg_label = int(bc.argmax())

    blocks: list[ColorBlock] = []
    for k, center in enumerate(centers):
        mask = (labels == k).astype(np.uint8) * 255
        if k == bg_label:
            # Split into the border-connected page and interior islands (art).
            count, comp = cv2.connectedComponents(mask, connectivity=4)
            edge = set(np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))) - {0}
            page = np.isin(comp, list(edge))
            b = block((page & (mask > 0)).astype(np.uint8) * 255, center, True)
            if b:
                blocks.append(b)
            b = block((~page & (mask > 0)).astype(np.uint8) * 255, center, False)
            if b:
                blocks.append(b)
        else:
            b = block(mask, center, False)
            if b:
                blocks.append(b)
    blocks.sort(key=lambda b: -b.area_pct)
    return blocks


# --------------------------------------------------------------------------- 1-9


@dataclass
class BlockPlan:
    thread_hex: str
    detected_hex: str
    area_pct: float
    shapes: list[Polygon]
    runs: list[list[Point]]


def _fit_all(groups: list[list[Polygon]], opts: Options) -> list[list[Polygon]]:
    """Fit and letter-space every block's polygons together, so all colors
    share one scale and one origin."""
    tags = [i for i, g in enumerate(groups) for _ in g]
    flat = [p for g in groups for p in g]
    width = opts.width_mm + 2 * INSET_MM
    max_h = opts.max_height_mm + 2 * INSET_MM if opts.max_height_mm else None
    flat = fit_to_size(flat, width, max_h)
    if opts.spacing_mm > 0:
        flat = fit_to_size(apply_letter_spacing(flat, opts.spacing_mm), width, max_h)
    out: list[list[Polygon]] = [[] for _ in groups]
    for t, p in zip(tags, flat):
        out[t].append(p)
    return out


def underlap(groups: list[list[Polygon]]) -> list[list[Polygon]]:
    """Extend each color under the colors sewn after it (by UNDERLAP_MM) so
    thread pull-in can't open a gap between neighbours. Never extends into
    the background or into colors already sewn."""
    unions = [shapely.unary_union(g) if g else Polygon() for g in groups]
    out = []
    for i, u in enumerate(unions):
        later = shapely.unary_union([v for v in unions[i + 1:]]) if i + 1 < len(unions) else Polygon()
        if later.is_empty or u.is_empty:
            out.append(polygons_of(u))
            continue
        # Grow outward, but keep only what lies on (or within the seam gap of)
        # a later color, so the extension is always covered by that color.
        grown = u.buffer(UNDERLAP_MM, join_style="mitre").intersection(later.buffer(SEAM_GAP_MM))
        out.append(polygons_of(shapely.make_valid(u.union(grown))))
    return out


def plan_design(image: np.ndarray, opts: Options) -> Design:
    if opts.colors <= 1:
        masks = [load_mask(image, opts.invert)]
        hexes = detected = [opts.thread_colors[0] if opts.thread_colors else "#ffffff"]
        areas = [100.0]
    else:
        blocks = [b for b in detect_colors(image, opts.colors) if opts.stitch_background or not b.is_background]
        if not blocks:
            raise DigitizeError("Every color was treated as background. Turn on 'stitch background'.")
        masks = [b.mask for b in blocks]
        detected = [b.hex for b in blocks]
        areas = [b.area_pct for b in blocks]
        hexes = list(opts.thread_colors) if opts.thread_colors else detected
        if len(hexes) != len(blocks):
            raise DigitizeError(f"Expected {len(blocks)} thread colors, got {len(hexes)}.")

    groups = [extract_polygons(m) for m in masks]
    if not any(groups):
        hint = "turning Invert off" if opts.invert else "turning Invert on"
        raise DigitizeError(f"No stitchable shapes found in the image. Try {hint}.")
    groups = _fit_all(groups, opts)
    min_feature = min_feature_height([p for g in groups for p in g])
    if len(groups) > 1:
        groups = underlap(groups)

    plans = []
    for g, thread, det, area in zip(groups, hexes, detected, areas):
        shapes = inset_shapes(g)
        if opts.pull_comp_mm > 0:
            # Grow back out across the board; the inset already opened the
            # holes and gaps, so this is a net (INSET - pull_comp) shrink.
            shapes = [q for p in shapes for q in polygons_of(p.buffer(opts.pull_comp_mm, join_style="mitre"))]
        shapes = order_center_out(simplify_shapes(shapes))
        if not shapes:
            continue
        runs = plan_stitches(shapes, opts.mode, opts.triple_run, opts.row_spacing_mm, opts.angle_deg, opts.underlay)
        plans.append(BlockPlan(thread, det, area, shapes, runs))
    if not plans:
        raise DigitizeError("Every shape is too small to stitch at this size. Try a larger size.")

    warnings = []
    if min_feature is not None and min_feature < SMALL_FEATURE_MM:
        warnings.append(
            f"Smallest text/feature is {min_feature:.1f} mm tall (under {SMALL_FEATURE_MM:g} mm): "
            "small lettering may not sew crisply."
        )
    return Design(
        shapes=[s for b in plans for s in b.shapes],
        runs=[r for b in plans for r in b.runs],
        min_feature_mm=min_feature,
        warnings=warnings,
        blocks=plans,
    )


# --------------------------------------------------------------------------- 10-11


def _toward(a: Point, b: Point, length: float) -> Point:
    d = _dist(a, b)
    t = min(length, d / 2) / d if d > 0 else 0
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def with_ties(run: list[Point]) -> list[Point]:
    """Add lock stitches: a back-and-forth at the start and end of a run so the
    thread doesn't pull out after a trim."""
    a, z = run[0], run[-1]
    return [a, _toward(a, run[1], TIE_MM), a] + run[1:] + [_toward(z, run[-2], TIE_MM), z]


def estimated_stitch_count(runs: list[list[Point]]) -> int:
    return sum(len(r) + 4 for r in runs)


def _units(p: Point) -> tuple[float, float]:
    return round(p[0] * DST_UNITS_PER_MM), round(p[1] * DST_UNITS_PER_MM)


def build_pattern(blocks: list[tuple[str, list[list[Point]]]]) -> pyembroidery.EmbPattern:
    """blocks: (thread hex, runs) per color, in sewing order. A COLOR_CHANGE
    (a stop, in DST) separates colors; TRIM ends every run."""
    pattern = pyembroidery.EmbPattern()
    for n, (color_hex, runs) in enumerate(blocks):
        thread = pyembroidery.EmbThread()
        thread.set_hex_color(color_hex)
        pattern.add_thread(thread)
        if n:
            pattern.add_command(pyembroidery.COLOR_CHANGE)
        for run in runs:
            pts = with_ties(run)
            pattern.add_stitch_absolute(pyembroidery.JUMP, *_units(pts[0]))
            for p in pts:
                pattern.add_stitch_absolute(pyembroidery.STITCH, *_units(p))
            pattern.add_command(pyembroidery.TRIM)
    pattern.add_command(pyembroidery.END)
    return pattern


def count_commands(pattern: pyembroidery.EmbPattern) -> tuple[int, int, int]:
    """(stitches, trims, color changes). Trims before the first stitch cut
    nothing and are ignored (the DST reader reports a long opening jump as a trim)."""
    stitches = trims = changes = 0
    for _, _, cmd in pattern.stitches:
        cmd &= pyembroidery.COMMAND_MASK
        if cmd == pyembroidery.STITCH:
            stitches += 1
        elif cmd == pyembroidery.TRIM and stitches:
            trims += 1
        elif cmd == pyembroidery.COLOR_CHANGE:
            changes += 1
    return stitches, trims, changes


def stitch_points(pattern: pyembroidery.EmbPattern) -> list[tuple[float, float, int]]:
    """(x_mm, y_mm, command) for every entry of a pattern."""
    return [(x / DST_UNITS_PER_MM, y / DST_UNITS_PER_MM, c & pyembroidery.COMMAND_MASK)
            for x, y, c in pattern.stitches]


def write_and_verify(pattern: pyembroidery.EmbPattern, path: Path) -> Verified:
    pyembroidery.write_dst(pattern, str(path))
    back = pyembroidery.read_dst(str(path))
    if back is None:
        raise DigitizeError("The DST file could not be read back.")
    stitches, trims, changes = count_commands(back)
    xy = [(x, y) for x, y, c in stitch_points(back) if c == pyembroidery.STITCH]
    if not xy:
        raise DigitizeError("The DST file has no stitches.")
    xs, ys = zip(*xy)
    return Verified(back, stitches, trims, changes, (min(xs), min(ys), max(xs), max(ys)))


# --------------------------------------------------------------------------- preview


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    h = color_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def render_preview(pattern: pyembroidery.EmbPattern, colors: list[str] | str = "#ffffff",
                   px_per_mm: float = 16, line_mm: float = 0.4, margin_mm: float = 4,
                   max_side_px: int = 8000) -> np.ndarray:
    """Draw stitches (not jumps) anti-aliased on a dark background, then
    downscale 50%. Each COLOR_CHANGE moves to the next color. Returns BGR."""
    colors = [colors] if isinstance(colors, str) else list(colors) or ["#ffffff"]
    polylines: list[tuple[int, list[tuple[float, float]]]] = []
    current: list[tuple[float, float]] = []
    ci = 0
    for x, y, cmd in stitch_points(pattern):
        if cmd == pyembroidery.STITCH:
            current.append((x, y))
            continue
        if current:
            polylines.append((ci, current))
            current = []
        if cmd == pyembroidery.COLOR_CHANGE:
            ci = min(ci + 1, len(colors) - 1)
    if current:
        polylines.append((ci, current))

    xy = [p for _, line in polylines for p in line] or [(0.0, 0.0)]
    xs, ys = zip(*xy)
    minx, miny = min(xs) - margin_mm, min(ys) - margin_mm
    w_mm, h_mm = max(xs) - minx + margin_mm, max(ys) - miny + margin_mm
    px_per_mm = min(px_per_mm, max_side_px / max(w_mm, h_mm))
    w, h = math.ceil(w_mm * px_per_mm), math.ceil(h_mm * px_per_mm)

    bgrs = [_hex_to_bgr(c) for c in colors]
    luminance = max((0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0]) / 255 for c in bgrs)
    background = (40, 36, 34) if luminance > 0.2 else (200, 204, 208)  # keep dark thread visible
    img = np.full((h, w, 3), background, np.uint8)

    shift = 4  # sub-pixel precision for cv2 drawing
    scale = px_per_mm * (1 << shift)
    thickness = max(1, round(line_mm * px_per_mm))
    for ci, line in polylines:
        if len(line) < 2:
            continue
        arr = np.array([((x - minx) * scale, (y - miny) * scale) for x, y in line], np.int32)
        cv2.polylines(img, [arr], False, bgrs[ci], thickness, cv2.LINE_AA, shift)
    return cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- orchestrator


FORMATS = {
    "dst": ("Tajima", pyembroidery.write_dst),
    "pes": ("Brother / Babylock", pyembroidery.write_pes),
    "jef": ("Janome", pyembroidery.write_jef),
    "exp": ("Melco / Bernina", pyembroidery.write_exp),
    "vp3": ("Husqvarna / Pfaff", pyembroidery.write_vp3),
}


@dataclass
class Result:
    design: Design
    verified: Verified
    dst_path: Path
    preview_path: Path
    files: dict[str, Path] = field(default_factory=dict)  # every requested format, incl. dst


def digitize_to_files(image: np.ndarray, opts: Options, out_dir: Path,
                      color_hex: str = "#ffffff", formats: tuple[str, ...] = ("dst",)) -> Result:
    if opts.colors <= 1 and not opts.thread_colors:
        opts = replace(opts, thread_colors=(color_hex,))
    design = plan_design(image, opts)
    too_many = (f"Design needs more than {MAX_STITCHES:,} stitches. "
                "Try a smaller size, fewer colors or Outline mode.")
    if estimated_stitch_count(design.runs) > MAX_STITCHES:
        raise DigitizeError(too_many)
    if not design.runs:
        raise DigitizeError("Nothing to stitch at this size. Try a larger size.")

    dst_path = out_dir / "design.dst"
    pattern = build_pattern([(b.thread_hex, b.runs) for b in design.blocks])
    verified = write_and_verify(pattern, dst_path)
    if verified.stitch_count > MAX_STITCHES:
        raise DigitizeError(too_many)
    files = {"dst": dst_path}
    for fmt in formats:
        if fmt != "dst" and fmt in FORMATS:
            files[fmt] = out_dir / f"design.{fmt}"
            FORMATS[fmt][1](pattern, str(files[fmt]))

    preview_path = out_dir / "preview.png"
    cv2.imwrite(str(preview_path), render_preview(verified.pattern, [b.thread_hex for b in design.blocks]))
    return Result(design, verified, dst_path, preview_path, files)
