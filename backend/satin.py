"""Satin (zigzag column) stitching for narrow shapes: lettering, strokes, outlines.

A professional digitizer never tatami-fills a 2 mm letter stroke; rows cut
across it leave fragments and a ragged edge. Satin lays each stitch from one
edge of the stroke to the other, following its centerline, which is what a
machine-embroidered letter looks like.

Steps:
  1. medial axis of the shape: Voronoi edges of densely sampled boundary
     points, keeping only the edges that lie inside the shape, merged into
     branches and pruned of the short spurs Voronoi throws at every corner
  2. along each branch, cast a perpendicular through the shape to find the two
     edge points, and alternate between them every SATIN_DENSITY_MM
  3. per branch: a center-run underlay out, then satin back, so the whole
     stroke sews in one go
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon

Point2 = tuple[float, float]

SATIN_DENSITY_MM = 0.2  # sample step along the column; stitches alternate sides, so 0.4 mm per edge
SATIN_MAX_WIDTH_MM = 6.0  # wider than this and it's an area: tatami instead
SATIN_MIN_LENGTH_MM = 1.2  # shorter skeletons (the dot of an i) are tatami'd
BOUNDARY_SAMPLE_MM = 0.25
CENTER_RUN_MM = 2.0  # underlay stitch length along the centerline


def inscribed_width(shape: Polygon, limit: float = 20.0) -> float:
    """Widest stroke in the shape (diameter of the largest inscribed circle)."""
    lo, hi = 0.0, limit
    for _ in range(16):
        mid = (lo + hi) / 2
        if shape.buffer(-mid / 2).is_empty:
            hi = mid
        else:
            lo = mid
    return lo


def is_narrow(shape: Polygon, max_width: float = SATIN_MAX_WIDTH_MM) -> bool:
    return shape.buffer(-max_width / 2).is_empty


def _sample_rings(shape: Polygon, step: float) -> list[Point2]:
    pts: list[Point2] = []
    for ring in [shape.exterior, *shape.interiors]:
        n = max(8, math.ceil(ring.length / step))
        for i in range(n):
            p = ring.interpolate(i / n, normalized=True)
            pts.append((p.x, p.y))
    return pts


def _branches(lines) -> list[LineString]:
    merged = shapely.line_merge(MultiLineString(lines)) if len(lines) > 1 else lines[0]
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def skeleton(shape: Polygon, prune_mm: float) -> list[LineString]:
    """Medial-axis branches of the shape (approximate, via Voronoi)."""
    pts = _sample_rings(shape, BOUNDARY_SAMPLE_MM)
    vd = shapely.voronoi_polygons(MultiPoint(pts), only_edges=True)
    inner = shape.buffer(-0.03)
    edges = [e for e in vd.geoms if e.length > 0 and inner.contains(e)]
    if not edges:
        return []
    branches = _branches(edges)
    # Prune leaf spurs: Voronoi runs an edge from the axis into every corner
    # of the boundary. Repeat, since pruning can expose a new short leaf.
    for _ in range(4):
        ends: dict[Point2, int] = {}
        for b in branches:
            for c in (b.coords[0], b.coords[-1]):
                k = (round(c[0], 4), round(c[1], 4))
                ends[k] = ends.get(k, 0) + 1
        keep = []
        for b in branches:
            a, z = (round(b.coords[0][0], 4), round(b.coords[0][1], 4)), (round(b.coords[-1][0], 4), round(b.coords[-1][1], 4))
            leaf = ends[a] == 1 or ends[z] == 1
            if leaf and b.length < prune_mm and len(branches) > 1:
                continue
            keep.append(b)
        if len(keep) == len(branches):
            break
        branches = _branches(keep) if keep else []
    return _extend_leaves([b.simplify(0.05) for b in branches if b.length > 0], shape)


LEAF_INSET_MM = 0.15  # how far short of the boundary an extended tip stops


def _extend_leaves(branches: list[LineString], shape: Polygon) -> list[LineString]:
    """The Voronoi axis ends where the stroke starts to round off, so satin
    built on it leaves every stroke tip bare. Push each free end straight on
    along its tangent until just short of the boundary."""
    ends: dict[Point2, int] = {}
    for b in branches:
        for c in (b.coords[0], b.coords[-1]):
            k = (round(c[0], 4), round(c[1], 4))
            ends[k] = ends.get(k, 0) + 1
    boundary = shape.boundary
    out = []
    for b in branches:
        coords = list(b.coords)
        for end in (0, -1):
            k = (round(coords[end][0], 4), round(coords[end][1], 4))
            if ends[k] != 1 or len(coords) < 2:
                continue
            tip, prev = coords[end], coords[1 if end == 0 else -2]
            tx, ty = tip[0] - prev[0], tip[1] - prev[1]
            norm = math.hypot(tx, ty)
            if norm < 1e-9:
                continue
            tx, ty = tx / norm, ty / norm
            ray = LineString([tip, (tip[0] + tx * 20, tip[1] + ty * 20)])
            hit = ray.intersection(boundary)
            if hit.is_empty:
                continue
            pts = [hit] if isinstance(hit, Point) else [g for g in getattr(hit, "geoms", []) if isinstance(g, Point)]
            if not pts:
                continue
            nearest = min(pts, key=lambda q: q.distance(Point(tip)))
            reach = max(0.0, nearest.distance(Point(tip)) - LEAF_INSET_MM)
            new_tip = (tip[0] + tx * reach, tip[1] + ty * reach)
            if end == 0:
                coords[0] = new_tip
            else:
                coords[-1] = new_tip
        out.append(LineString(coords))
    return out


def _cross_section(shape: Polygon, p: Point2, normal: Point2, half: float) -> tuple[Point2, Point2] | None:
    """Where a perpendicular through p meets the two edges of the stroke."""
    ray = LineString([(p[0] - normal[0] * half, p[1] - normal[1] * half),
                      (p[0] + normal[0] * half, p[1] + normal[1] * half)])
    cut = ray.intersection(shape)
    pieces = list(cut.geoms) if hasattr(cut, "geoms") else [cut]
    here = Point(p)
    best = None
    for g in pieces:
        if isinstance(g, LineString) and not g.is_empty and g.distance(here) < 1e-6:
            best = g
            break
    if best is None:
        return None
    a, b = best.coords[0], best.coords[-1]
    # Consistent side order: a on the -normal side, b on the +normal side.
    if (a[0] - p[0]) * normal[0] + (a[1] - p[1]) * normal[1] > 0:
        a, b = b, a
    return (a[0], a[1]), (b[0], b[1])


RAY_STRETCH = 1.6  # a cross-section may reach this multiple of the local half-width


def zigzag(shape: Polygon, branch: LineString, density: float = SATIN_DENSITY_MM,
           max_width: float = SATIN_MAX_WIDTH_MM) -> list[Point2]:
    """Satin needle points along one skeleton branch. Each cross-section is
    capped near the local stroke width so that at a junction (or where two
    strokes run close) it can't shoot across into the neighbouring stroke."""
    n = max(2, math.ceil(branch.length / density))
    boundary = shape.boundary
    out: list[Point2] = []
    for i in range(n + 1):
        d = branch.length * i / n
        p = branch.interpolate(d)
        q0, q1 = branch.interpolate(max(0.0, d - 0.1)), branch.interpolate(min(branch.length, d + 0.1))
        tx, ty = q1.x - q0.x, q1.y - q0.y
        norm = math.hypot(tx, ty) or 1.0
        normal = (-ty / norm, tx / norm)
        half = min(max_width / 2, boundary.distance(p) * RAY_STRETCH + 0.2)
        cs = _cross_section(shape, (p.x, p.y), normal, half)
        if cs is None:
            continue
        out.append(cs[i % 2])
    return out


def center_run(branch: LineString, step: float = CENTER_RUN_MM) -> list[Point2]:
    n = max(1, math.ceil(branch.length / step))
    return [(pt.x, pt.y) for pt in (branch.interpolate(branch.length * i / n) for i in range(n + 1))]


def satin_plan(shape: Polygon, start: Point2, density: float = SATIN_DENSITY_MM,
               max_width: float = SATIN_MAX_WIDTH_MM) -> list[list[Point2]] | None:
    """Stitch runs (center-run underlay + satin) for a narrow shape, or None
    if it isn't a good satin candidate. Each run is contiguous; the caller
    joins runs with walk stitches."""
    width = inscribed_width(shape, max_width + 1)
    branches = skeleton(shape, prune_mm=max(0.8, 0.8 * width))
    if not branches or sum(b.length for b in branches) < SATIN_MIN_LENGTH_MM:
        return None
    runs: list[list[Point2]] = []
    here = start
    remaining = list(branches)
    while remaining:
        # Nearest branch end to where we are; sew it from that end.
        def dist_to(b: LineString) -> tuple[float, bool]:
            d0 = math.dist(here, b.coords[0])
            d1 = math.dist(here, b.coords[-1])
            return (d0, False) if d0 <= d1 else (d1, True)
        b = min(remaining, key=lambda x: dist_to(x)[0])
        remaining.remove(b)
        if dist_to(b)[1]:
            b = LineString(b.coords[::-1])
        under = center_run(b)
        sat = zigzag(shape, LineString(b.coords[::-1]), density, max_width)  # back toward the start
        if len(sat) < 2:
            continue
        runs.append(under + sat)
        here = sat[-1]
    return runs or None
