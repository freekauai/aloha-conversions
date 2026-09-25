import math

import cv2
import shapely
import numpy as np
import pyembroidery
import pytest
from shapely.geometry import LineString, Point

from backend import digitize as dz


def rectangle_image(w=400, h=200, pad=50):
    img = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    img[pad:pad + h, pad:pad + w] = 255
    return img


def letter_image(text="B"):
    img = np.zeros((400, 360), np.uint8)
    cv2.putText(img, text, (40, 340), cv2.FONT_HERSHEY_DUPLEX, 12, 255, 40)
    return img


def run(image, opts, tmp_path):
    return dz.digitize_to_files(image, opts, tmp_path)


def stitch_segments(pattern):
    """Consecutive needle-to-needle stitch lines (jumps and trims break the line)."""
    segs, prev = [], None
    for x, y, cmd in dz.stitch_points(pattern):
        if cmd == pyembroidery.STITCH:
            if prev is not None and prev != (x, y):
                segs.append((prev, (x, y)))
            prev = (x, y)
        else:
            prev = None
    return segs


def test_white_rectangle_is_one_shape_with_expected_stitch_count(tmp_path):
    result = run(rectangle_image(), dz.Options(width_mm=100), tmp_path)
    assert len(result.design.shapes) == 1

    # 100 x 50 mm: ~119 tatami rows of ~34 needle points, plus a ~290 mm underlay lap.
    rows = 50 / dz.FILL_ROW_SPACING_MM
    expected = rows * (100 / dz.FILL_STITCH_MM + 1) + 290 / dz.UNDERLAY_STITCH_MM
    assert 0.8 * expected < result.verified.stitch_count < 1.25 * expected
    assert result.verified.trim_count == 1  # just the end-of-shape trim


def test_letter_b_has_two_holes_and_no_stitches_inside_them(tmp_path):
    result = run(letter_image("B"), dz.Options(width_mm=40), tmp_path)
    assert len(result.design.shapes) == 1
    shape = result.design.shapes[0]
    assert len(shape.interiors) == 2

    # Shrink the holes by the DST's 0.1 mm rounding so points exactly on the edge don't count.
    rounding = 0.05 * math.sqrt(2) + 0.01
    points = [Point(x, y) for x, y, c in dz.stitch_points(result.verified.pattern)
              if c == pyembroidery.STITCH]
    assert points
    for ring in shape.interiors:
        hole = dz.Polygon(ring)
        assert not any(hole.buffer(-rounding).contains(p) for p in points)
        # Row-to-row stitches may chord across a round counter's edge by at most
        # the travel tolerance; nothing may actually cross a hole.
        deep = hole.buffer(-(dz.TRAVEL_TOLERANCE_MM + rounding))
        assert not any(deep.intersects(LineString(s)) for s in stitch_segments(result.verified.pattern))


def test_two_separated_squares_have_a_trim_between_them(tmp_path):
    img = np.zeros((300, 800), np.uint8)
    img[50:250, 50:250] = 255
    img[50:250, 550:750] = 255
    result = run(img, dz.Options(width_mm=80, spacing_mm=0), tmp_path)
    assert len(result.design.shapes) == 2

    side = None  # which square (-1 left, +1 right) the last stitch was in
    trimmed = False
    crossings = 0
    for x, _, cmd in dz.stitch_points(result.verified.pattern):
        if cmd == pyembroidery.TRIM:
            trimmed = True
        elif cmd == pyembroidery.STITCH:
            here = -1 if x < 0 else 1
            if side is not None and here != side:
                crossings += 1
                assert trimmed, "moved between squares without a trim"
            if side != here:
                trimmed = False
            side = here
    assert crossings >= 1
    assert result.verified.trim_count >= 1
    # And no stitch line crosses the empty gap between the squares.
    gap = dz.Polygon([(-15, -100), (15, -100), (15, 100), (-15, 100)])
    assert not any(gap.intersects(LineString(s)) for s in stitch_segments(result.verified.pattern))


@pytest.mark.parametrize("width_mm", [40, 76.2, 107.95])
def test_output_width_matches_request(tmp_path, width_mm):
    img = np.zeros((300, 900), np.uint8)
    cv2.putText(img, "KAUAI", (20, 230), cv2.FONT_HERSHEY_DUPLEX, 6, 255, 24)
    result = run(img, dz.Options(width_mm=width_mm), tmp_path)
    assert abs(result.verified.width_mm - width_mm) <= 0.5


def test_height_is_clamped_to_preset_max(tmp_path):
    width, max_h = dz.resolve_size("hat")
    tall = rectangle_image(w=200, h=400)
    result = run(tall, dz.Options(width_mm=width, max_height_mm=max_h), tmp_path)
    assert result.verified.height_mm <= max_h + 0.5
    assert abs(result.verified.height_mm - max_h) <= 0.5


def test_subject_auto_stitches_the_artwork_not_the_page(tmp_path):
    black_on_white = 255 - rectangle_image()
    white_on_black = rectangle_image()
    for img, expect in ((black_on_white, "dark"), (white_on_black, "light")):
        assert dz.detect_subject(img) == expect
        result = run(img, dz.Options(width_mm=100), tmp_path)  # subject="auto"
        assert result.design.subject == expect
        assert len(result.design.shapes) == 1 and len(result.design.shapes[0].interiors) == 0
    # Forcing "light" on the black-on-white page stitches the page: a frame with a hole.
    plain = dz.plan_design(black_on_white, dz.Options(width_mm=100, subject="light"))
    assert [len(s.interiors) for s in plain.shapes] == [1]
    # Legacy invert=True still means "dark".
    assert run(black_on_white, dz.Options(width_mm=100, invert=True), tmp_path).design.subject == "dark"


def test_subject_auto_on_transparent_png():
    img = np.zeros((200, 400, 4), np.uint8)
    cv2.putText(img, "HI", (40, 150), cv2.FONT_HERSHEY_DUPLEX, 4, (20, 20, 20, 255), 12)  # dark art, clear page
    assert dz.detect_subject(img) == "dark"
    img[:, :, :3] = 255; img[:, :, :3][img[:, :, 3] == 0] = 0
    assert dz.detect_subject(img) == "light"  # white art, clear page


def test_blank_image_raises(tmp_path):
    with pytest.raises(dz.DigitizeError, match="stitching the"):
        run(np.zeros((100, 100), np.uint8), dz.Options(width_mm=50), tmp_path)


def test_outline_mode_only_follows_edges(tmp_path):
    fill = run(rectangle_image(), dz.Options(width_mm=100), tmp_path)
    outline = run(rectangle_image(), dz.Options(width_mm=100, mode="outline"), tmp_path)
    assert outline.verified.stitch_count < fill.verified.stitch_count / 10
    triple = run(rectangle_image(), dz.Options(width_mm=100, mode="outline", triple_run=True), tmp_path)
    assert triple.verified.stitch_count > 2.5 * outline.verified.stitch_count


def test_stitch_lengths_stay_within_limits(tmp_path):
    result = run(letter_image("R"), dz.Options(width_mm=60), tmp_path)
    lengths = [math.dist(a, b) for a, b in stitch_segments(result.verified.pattern)]
    assert max(lengths) <= dz.FILL_STITCH_MM + 0.15  # 0.1 mm DST rounding


def test_too_many_stitches_is_rejected(tmp_path):
    with pytest.raises(dz.DigitizeError, match="smaller size"):
        run(rectangle_image(), dz.Options(width_mm=500), tmp_path)


def test_small_lettering_warning(tmp_path):
    img = np.zeros((200, 1400), np.uint8)
    cv2.putText(img, "SMALL TEXT", (10, 150), cv2.FONT_HERSHEY_DUPLEX, 4, 255, 14)
    small = run(img, dz.Options(width_mm=50), tmp_path)
    assert small.design.warnings and "may not sew crisply" in small.design.warnings[0]
    big = run(letter_image("B"), dz.Options(width_mm=40), tmp_path)
    assert big.design.warnings == []


def test_small_second_line_warns_but_dot_of_i_does_not(tmp_path):
    img = np.zeros((460, 1200), np.uint8)
    cv2.putText(img, "Kiki", (80, 250), cv2.FONT_HERSHEY_DUPLEX, 8, 255, 30)
    big_only = dz.plan_design(img, dz.Options(width_mm=50))
    assert big_only.warnings == []  # the i dots belong to their letters

    cv2.putText(img, "SURF SHOP", (80, 420), cv2.FONT_HERSHEY_DUPLEX, 2, 255, 8)
    two_lines = dz.plan_design(img, dz.Options(width_mm=50))
    assert two_lines.warnings and two_lines.min_feature_mm < 6


def test_letter_spacing_moves_whole_glyphs(tmp_path):
    img = np.zeros((300, 400), np.uint8)
    cv2.putText(img, "ii", (40, 250), cv2.FONT_HERSHEY_DUPLEX, 8, 255, 30)
    polys = dz.extract_polygons(dz.load_mask(img))
    assert len(polys) == 4  # two stems, two dots
    assert len(dz.glyph_clusters(polys)) == 2


def test_preview_renders(tmp_path):
    result = run(letter_image("B"), dz.Options(width_mm=40), tmp_path)
    img = cv2.imread(str(result.preview_path))
    assert img is not None and img.shape[1] > 100
    # 16 px/mm, halved, plus 4 mm margins each side
    assert abs(img.shape[1] - (result.verified.width_mm + 8) * 8) < 3


@pytest.mark.parametrize("name", list(dz.PRESETS))
def test_every_preset_resolves_and_caps_height(name):
    width_mm, max_h = dz.resolve_size(name)
    p = dz.PRESETS[name]
    assert width_mm == pytest.approx(p.width_in * dz.MM_PER_INCH)
    assert max_h == pytest.approx(p.max_height_in * dz.MM_PER_INCH)


def test_custom_height_caps_tall_designs(tmp_path):
    width_mm, max_h = dz.resolve_size("custom", 4, 1.5)
    result = run(rectangle_image(w=200, h=400), dz.Options(width_mm=width_mm, max_height_mm=max_h), tmp_path)
    assert abs(result.verified.height_mm - 1.5 * dz.MM_PER_INCH) <= 0.5


def test_full_front_fill_fits_under_the_cap(tmp_path):
    img = np.zeros((500, 1400), np.uint8)
    cv2.putText(img, "ALOHA", (40, 400), cv2.FONT_HERSHEY_DUPLEX, 12, 255, 45)
    width_mm, max_h = dz.resolve_size("full_front")
    result = run(img, dz.Options(width_mm=width_mm, max_height_mm=max_h), tmp_path)
    assert result.verified.stitch_count <= dz.MAX_STITCHES


# --------------------------------------------------------------------------- multi-color


def three_color_image():
    """Red disc and a blue box with green text, on a white page."""
    img = np.full((400, 1000, 3), 255, np.uint8)
    cv2.circle(img, (300, 200), 150, (30, 60, 200), -1)
    cv2.rectangle(img, (500, 80), (900, 320), (180, 80, 0), -1)
    cv2.putText(img, "KT", (560, 280), cv2.FONT_HERSHEY_DUPLEX, 5, (30, 200, 60), 18)
    return img


def test_detect_colors_finds_blocks_and_background():
    blocks = dz.detect_colors(three_color_image(), 4)
    assert [b.is_background for b in blocks] == [True, False, False, False]
    assert blocks[0].hex == "#ffffff"
    assert [round(b.area_pct) for b in blocks] == sorted([round(b.area_pct) for b in blocks], reverse=True)
    # Deterministic: a second call (as the digitize request makes) matches.
    again = dz.detect_colors(three_color_image(), 4)
    assert [b.hex for b in again] == [b.hex for b in blocks]


def test_multicolor_sews_largest_first_with_color_changes(tmp_path):
    result = run(three_color_image(), dz.Options(width_mm=100, colors=4), tmp_path)
    blocks = result.design.blocks
    assert len(blocks) == 3  # white background skipped
    assert [b.area_pct for b in blocks] == sorted([b.area_pct for b in blocks], reverse=True)
    assert result.verified.color_changes == 2

    # Stitches between the two color changes all belong to the second block.
    seen, ci = [], 0
    for x, y, c in dz.stitch_points(result.verified.pattern):
        if c == pyembroidery.COLOR_CHANGE:
            ci += 1
        elif c == pyembroidery.STITCH:
            seen.append((ci, x, y))
    assert {ci for ci, _, _ in seen} == {0, 1, 2}
    red = shapely.unary_union(blocks[1].shapes).buffer(0.3)
    assert all(red.covers(Point(x, y)) for ci, x, y in seen if ci == 1)


def test_underlap_extends_lower_color_under_upper(tmp_path):
    result = run(three_color_image(), dz.Options(width_mm=100, colors=4), tmp_path)
    blue, green = result.design.blocks[0], result.design.blocks[2]
    overlap = shapely.unary_union(blue.shapes).intersection(shapely.unary_union(green.shapes))
    perimeter = shapely.unary_union(green.shapes).length
    assert 0.15 <= overlap.area / perimeter <= 0.45  # ~0.3 mm band under the green edge
    # But blue never spills into the page around it.
    assert shapely.unary_union(blue.shapes).bounds[0] > shapely.unary_union(green.shapes).bounds[0] - 100


def test_thread_colors_override_detected(tmp_path):
    result = run(three_color_image(), dz.Options(width_mm=100, colors=4,
                                                 thread_colors=("#111111", "#222222", "#333333")), tmp_path)
    assert [b.thread_hex for b in result.design.blocks] == ["#111111", "#222222", "#333333"]
    assert result.design.blocks[0].detected_hex != "#111111"
    with pytest.raises(dz.DigitizeError, match="thread colors"):
        run(three_color_image(), dz.Options(width_mm=100, colors=4, thread_colors=("#111111",)), tmp_path)


def test_stitch_background_includes_the_page(tmp_path):
    result = run(three_color_image(), dz.Options(width_mm=100, colors=4, stitch_background=True), tmp_path)
    assert len(result.design.blocks) == 4
    assert result.design.blocks[0].detected_hex == "#ffffff"


def test_transparent_png_has_no_background_block():
    img = np.zeros((300, 600, 4), np.uint8)
    cv2.circle(img, (150, 150), 100, (30, 60, 200, 255), -1)
    cv2.rectangle(img, (350, 50), (550, 250), (180, 80, 0, 255), -1)
    blocks = dz.detect_colors(img, 2)
    assert len(blocks) == 2 and not any(b.is_background for b in blocks)


def test_multicolor_preview_uses_each_thread(tmp_path):
    result = run(three_color_image(), dz.Options(width_mm=100, colors=4,
                                                 thread_colors=("#ff0000", "#00ff00", "#0000ff")), tmp_path)
    img = cv2.imread(str(result.preview_path))
    px = img.reshape(-1, 3)
    for bgr in [(0, 0, 255), (0, 255, 0), (255, 0, 0)]:
        assert (np.abs(px.astype(int) - bgr).sum(axis=1) < 60).sum() > 500


def test_page_color_inside_the_art_is_kept():
    """White page, blue box, white text inside the box: the page is skipped
    but the white text stays, as its own block of the same color."""
    img = np.full((400, 1000, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (950, 350), (200, 60, 20), -1)
    cv2.putText(img, "HI", (300, 300), cv2.FONT_HERSHEY_DUPLEX, 7, (255, 255, 255), 30)
    blocks = dz.detect_colors(img, 3)
    kinds = [(b.hex, b.is_background) for b in blocks]
    assert ("#ffffff", True) in kinds and ("#ffffff", False) in kinds
    text = next(b for b in blocks if b.hex == "#ffffff" and not b.is_background)
    assert 1 < text.area_pct < 10


def test_opaque_alpha_channel_still_finds_the_page():
    img = cv2.cvtColor(three_color_image(), cv2.COLOR_BGR2BGRA)  # alpha = 255 everywhere
    blocks = dz.detect_colors(img, 4)
    assert blocks[0].is_background and blocks[0].hex == "#ffffff"


def test_antialiased_edges_do_not_become_a_color():
    img = np.full((400, 1000, 3), 255, np.uint8)
    cv2.circle(img, (500, 200), 150, (200, 60, 20), -1, lineType=cv2.LINE_AA)
    blocks = dz.detect_colors(img, 4)
    assert len(blocks) == 2  # page + disc; the soft edge merged away


# --------------------------------------------------------------------------- sewing controls


def fill_directions(pattern):
    """Unit direction of every fill-length stitch, so we can read the fill angle back."""
    dirs = []
    for (ax, ay), (bx, by) in stitch_segments(pattern):
        d = math.hypot(bx - ax, by - ay)
        if d > 2.6:  # fill rows run to 3.0 mm; underlay is 2.5, ties shorter
            dirs.append(((bx - ax) / d, (by - ay) / d))
    return dirs


def test_fill_angle_rotates_the_rows(tmp_path):
    flat = run(rectangle_image(), dz.Options(width_mm=60, angle_deg=0), tmp_path)
    tilted = run(rectangle_image(), dz.Options(width_mm=60, angle_deg=45), tmp_path)
    assert all(abs(dy) < 0.05 for _, dy in fill_directions(flat.verified.pattern))
    diag = [abs(abs(dx) - abs(dy)) < 0.1 for dx, dy in fill_directions(tilted.verified.pattern)]
    assert sum(diag) > 0.9 * len(diag)
    assert abs(tilted.verified.width_mm - 60) <= 0.5  # angle doesn't change the size


def test_density_changes_stitch_count(tmp_path):
    light = run(rectangle_image(), dz.Options(width_mm=60, row_spacing_mm=0.50), tmp_path)
    dense = run(rectangle_image(), dz.Options(width_mm=60, row_spacing_mm=0.35), tmp_path)
    assert 1.25 < dense.verified.stitch_count / light.verified.stitch_count < 1.6


def test_underlay_types(tmp_path):
    none = run(rectangle_image(), dz.Options(width_mm=60, underlay="none"), tmp_path)
    contour = run(rectangle_image(), dz.Options(width_mm=60, underlay="contour"), tmp_path)
    full = run(rectangle_image(), dz.Options(width_mm=60, underlay="full"), tmp_path)
    assert none.verified.stitch_count < contour.verified.stitch_count < full.verified.stitch_count
    # "full" adds sparse vertical rows: some stitch directions should be vertical.
    vertical = [abs(dx) < 0.05 for dx, _ in fill_directions(full.verified.pattern)]
    assert any(vertical) and not all(vertical)


def test_pull_compensation_grows_shapes(tmp_path):
    base = run(rectangle_image(), dz.Options(width_mm=60), tmp_path)
    comp = run(rectangle_image(), dz.Options(width_mm=60, pull_comp_mm=0.3), tmp_path)
    assert comp.design.shapes[0].area > base.design.shapes[0].area
    assert comp.verified.width_mm - base.verified.width_mm == pytest.approx(0.6, abs=0.2)


def test_extra_formats_are_written_and_readable(tmp_path):
    result = dz.digitize_to_files(three_color_image(), dz.Options(width_mm=60, colors=4), tmp_path,
                                  formats=("dst", "pes", "jef", "exp", "vp3"))
    assert set(result.files) == {"dst", "pes", "jef", "exp", "vp3"}
    for fmt, path in result.files.items():
        assert path.stat().st_size > 100
        back = pyembroidery.read(str(path))
        assert back is not None and back.count_stitches() > 100, fmt
    # PES carries colors: three threads for three blocks.
    pes = pyembroidery.read(str(result.files["pes"]))
    assert len(pes.threadlist) == 3


# --------------------------------------------------------------------------- satin


def test_narrow_stroke_gets_satin_across_its_width(tmp_path):
    """A 2 mm x 40 mm bar: satin stitches run edge to edge (~2 mm long), not
    along 0.42 mm tatami rows, and the column reaches both ends of the bar."""
    img = np.zeros((300, 900), np.uint8)
    img[140:160, 50:850] = 255  # 20 px tall bar -> ~2.2 mm at 100 mm wide... scaled below
    result = run(img, dz.Options(width_mm=40), tmp_path)
    shape = result.design.shapes[0]
    width = shape.bounds[3] - shape.bounds[1]
    assert width < 3
    lengths = [math.dist(a, b) for a, b in stitch_segments(result.verified.pattern)]
    across = [l for l in lengths if abs(l - width) < 0.3]
    assert len(across) > 0.6 * len(lengths)  # most stitches span the stroke
    assert abs(result.verified.width_mm - 40) <= 0.6  # satin reaches the bar's ends
    # Turned off, the same bar is tatami: mostly short row stitches.
    plain = run(img, dz.Options(width_mm=40, satin_max_mm=0), tmp_path)
    lengths = [math.dist(a, b) for a, b in stitch_segments(plain.verified.pattern)]
    assert sum(1 for l in lengths if abs(l - width) < 0.3) < 0.2 * len(lengths)


def test_wide_shape_stays_tatami(tmp_path):
    result = run(rectangle_image(), dz.Options(width_mm=60), tmp_path)  # 60 x 30 mm block
    assert not dz.satin_mod.is_narrow(result.design.shapes[0])
    dirs = fill_directions(result.verified.pattern)
    assert len(dirs) > 100 and all(abs(dy) < 0.05 for _, dy in dirs)


def test_ring_letter_satin_stays_inside_and_ends_at_origin(tmp_path):
    result = run(letter_image("O"), dz.Options(width_mm=20), tmp_path)  # ~4.5 mm wall
    shape = result.design.shapes[0]
    assert dz.satin_mod.is_narrow(shape)
    pts = [(x, y) for x, y, c in dz.stitch_points(result.verified.pattern) if c == pyembroidery.STITCH]
    grown = shape.buffer(0.2)
    assert all(grown.covers(Point(p)) for p in pts)
    # Last movement is the jump home, so the next design starts centered.
    x, y, cmd = dz.stitch_points(result.verified.pattern)[-2]
    assert (x, y) == (0.0, 0.0)
