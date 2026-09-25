"""Real thread charts, from the tables pyembroidery ships for the machine
formats (Brother PES/PEC and Janome JEF). Names and catalog numbers are the
manufacturers' own, so an embroiderer can match them."""

from __future__ import annotations

import math

from pyembroidery import EmbThreadJef, EmbThreadPec

CHARTS: dict[str, dict] = {}


def _load(key: str, label: str, module) -> None:
    seen = set()
    threads = []
    for t in module.get_thread_set():
        if t is None or not t.description or t.hex_color() in seen:
            continue
        seen.add(t.hex_color())
        threads.append({"num": str(t.catalog_number), "name": t.description, "hex": t.hex_color()})
    CHARTS[key] = {"label": label, "threads": threads}


_load("brother", "Brother / Babylock", EmbThreadPec)
_load("janome", "Janome", EmbThreadJef)


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lab(h: str) -> tuple[float, float, float]:
    """sRGB hex -> CIE L*a*b* (D65), so "closest" means closest to the eye."""
    def lin(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in _rgb(h))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.0
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


HUE_WEIGHT = 3.0  # a thread of the right hue beats one of the right brightness


def _distance(a: str, b: str) -> float:
    """Lab lightness and chroma, plus a hue term that only counts when both
    colors are chromatic (so grays don't get pulled toward a random hue).
    Plain Lab distance matched a bright blue to purple; this doesn't."""
    l1, a1, b1 = _lab(a)
    l2, a2, b2 = _lab(b)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    dh = abs(math.atan2(b1, a1) - math.atan2(b2, a2))
    dh = min(dh, 2 * math.pi - dh)
    return (l1 - l2) ** 2 + (c1 - c2) ** 2 + (HUE_WEIGHT * math.sqrt(c1 * c2) * dh) ** 2


def nearest(chart: str, hex_color: str) -> dict:
    """Closest chart thread to a color."""
    return min(CHARTS[chart]["threads"], key=lambda t: _distance(hex_color, t["hex"]))
