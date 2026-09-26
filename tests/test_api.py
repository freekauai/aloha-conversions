import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture
def client():
    main._hits.clear()  # the limiter is process-wide; each test starts fresh
    with TestClient(main.app) as c:
        yield c


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def text_png():
    img = np.zeros((200, 700), np.uint8)
    cv2.putText(img, "SURF", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 5, 255, 20)
    return png_bytes(img)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Image to DST" in r.text


def test_digitize_returns_files_inline(client):
    r = client.post("/api/digitize", files={"file": ("my logo.png", text_png(), "image/png")},
                    data={"preset": "hat", "color": "#ff0000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(body["stats"]["width_in"] - 4.25) <= 0.02
    assert body["stats"]["stitches"] > 0 and body["stats"]["trims"] >= 1
    assert body["filename"] == "my-logo"

    dst = base64.b64decode(body["dst_base64"])
    assert len(dst) > 512 and dst.startswith(b"LA:")  # 512-byte DST header
    png = base64.b64decode(body["preview_png_base64"])
    assert png[:4] == b"\x89PNG"


def test_custom_width(client):
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")},
                    data={"preset": "custom", "width_in": "2"})
    assert r.status_code == 200, r.text
    assert abs(r.json()["stats"]["width_in"] - 2) <= 0.02


def test_rejects_non_images(client):
    r = client.post("/api/digitize", files={"file": ("a.gif", b"GIF89a....", "image/gif")})
    assert r.status_code == 415


def test_upload_limit_is_configurable(monkeypatch, client):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")})
    assert r.status_code == 413


def test_rejects_oversized_dimensions(client):
    big = png_bytes(np.zeros((10, 4001), np.uint8))
    r = client.post("/api/digitize", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413


def test_jpeg_dimensions_are_read_from_header():
    ok, buf = cv2.imencode(".jpg", np.zeros((123, 456, 3), np.uint8))
    assert main.image_dimensions(buf.tobytes()) == (456, 123)


def test_blank_image_gives_helpful_error(client):
    r = client.post("/api/digitize", files={"file": ("a.png", png_bytes(np.zeros((50, 50), np.uint8)), "image/png")})
    assert r.status_code == 422 and "stitching the" in r.json()["detail"]


def test_presets_endpoint_lists_every_preset(client):
    body = client.get("/api/presets").json()
    assert {p["id"] for p in body["presets"]} == set(main.dz.PRESETS)
    assert body["custom"]["max_width_in"] == main.MAX_WIDTH_IN


def test_cors_allows_kauaitoday(client):
    r = client.options("/api/digitize", headers={"Origin": "https://kauaitoday.info",
                                                 "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "https://kauaitoday.info"


def color_png():
    img = np.full((300, 800, 3), 255, np.uint8)
    cv2.circle(img, (200, 150), 110, (30, 60, 200), -1)
    cv2.rectangle(img, (400, 40), (760, 260), (180, 80, 0), -1)
    return png_bytes(img)


def test_colors_endpoint_and_multicolor_digitize(client):
    r = client.post("/api/colors", files={"file": ("a.png", color_png(), "image/png")}, data={"colors": "3"})
    assert r.status_code == 200, r.text
    blocks = r.json()["blocks"]
    assert len(blocks) == 3 and blocks[0]["is_background"]

    threads = json.dumps(["#123456", "#abcdef"])
    r = client.post("/api/digitize", files={"file": ("a.png", color_png(), "image/png")},
                    data={"preset": "left_chest", "colors": "3", "thread_colors": threads})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stats"]["color_changes"] == 1
    assert [b["thread"] for b in body["blocks"]] == ["#123456", "#abcdef"]
    assert all(b["stitches"] > 0 for b in body["blocks"])


def test_bad_thread_colors_rejected(client):
    r = client.post("/api/digitize", files={"file": ("a.png", color_png(), "image/png")},
                    data={"colors": "3", "thread_colors": '["red"]'})
    assert r.status_code == 422
    r = client.post("/api/digitize", files={"file": ("a.png", color_png(), "image/png")},
                    data={"colors": "9"})
    assert r.status_code == 422


def test_rate_limit_blocks_after_the_window_fills(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 3)
    main._hits.clear()
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    for _ in range(3):
        r = client.post("/api/colors", files={"file": ("a.png", color_png(), "image/png")},
                        data={"colors": "2"}, headers=headers)
        assert r.status_code == 200
    r = client.post("/api/colors", files={"file": ("a.png", color_png(), "image/png")},
                    data={"colors": "2"}, headers=headers)
    assert r.status_code == 429 and "Retry-After" in r.headers
    # A different IP is unaffected, and GETs are never limited.
    r = client.post("/api/colors", files={"file": ("a.png", color_png(), "image/png")},
                    data={"colors": "2"}, headers={"x-forwarded-for": "198.51.100.4"})
    assert r.status_code == 200
    assert client.get("/api/presets", headers=headers).status_code == 200


def test_rate_limit_window_slides():
    main._hits.clear()
    for t in range(main.RATE_LIMIT):
        assert main.rate_limited("1.2.3.4", now=1000 + t) == 0
    assert main.rate_limited("1.2.3.4", now=1000 + main.RATE_LIMIT) > 0
    assert main.rate_limited("1.2.3.4", now=1000 + main.RATE_WINDOW_SECONDS + 1) == 0


def test_threads_and_sewing_endpoints(client):
    t = client.get("/api/threads").json()["charts"]
    assert {"brother", "janome"} <= set(t)
    assert all(k in t["brother"]["threads"][0] for k in ("num", "name", "hex"))
    s = client.get("/api/sewing").json()
    assert {f["id"] for f in s["fabrics"]} >= {"cap", "tee", "fleece"}
    assert [f["id"] for f in s["formats"]] == ["dst", "pes", "jef", "exp", "vp3"]


def test_fabric_preset_and_formats_in_digitize(client):
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")},
                    data={"preset": "hat", "fabric": "fleece", "angle_deg": "30", "formats": "dst,pes,jef"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sewing"] == {"row_spacing_mm": 0.40, "angle_deg": 30, "underlay": "full",
                              "pull_comp_mm": 0.30, "fabric": "fleece", "satin_max_mm": 6.0, "outline_width_mm": 1.5}
    assert set(body["files_base64"]) == {"dst", "pes", "jef"}
    assert body["threads"]["brother"][0]["name"]


def test_sewing_overrides_and_validation(client):
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")},
                    data={"fabric": "cap", "density": "dense", "underlay": "none", "pull_comp_mm": "0.1"})
    assert r.status_code == 200 and r.json()["sewing"]["row_spacing_mm"] == 0.35
    assert r.json()["sewing"]["underlay"] == "none"
    for bad in ({"fabric": "silk"}, {"density": "extreme"}, {"underlay": "lots"},
                {"pull_comp_mm": "2"}, {"formats": "dst,xyz"}, {"angle_deg": "120"}):
        r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")}, data=bad)
        assert r.status_code == 422, bad


def test_nearest_thread_is_perceptual():
    from backend import threads as th
    names = {hx: th.nearest("brother", hx)["name"] for hx in ("#fcd23d", "#ffffff", "#1e50d6", "#c8102e", "#00843d", "#111111")}
    assert names == {"#fcd23d": "Harvest Gold", "#ffffff": "White", "#1e50d6": "Ultramarine",
                     "#c8102e": "Red", "#00843d": "Emerald Green", "#111111": "Black"}
    assert "blue" in th.nearest("janome", "#1e50d6")["name"].lower()


def test_stats_are_reported_server_side(client, monkeypatch):
    sent = []

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        sent.append((req.full_url, req.headers, json.loads(req.data)))
        return FakeResp()

    monkeypatch.setattr(main, "STATS_SECRET", "s3cret")
    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")},
                    data={"preset": "hat", "fabric": "tee", "formats": "dst,pes"})
    assert r.status_code == 200 and r.json()["stats_recorded"] is True
    url, headers, body = sent[-1]
    assert url == main.STATS_URL and headers["X-aloha-secret"] == "s3cret"
    assert body["kind"] == "aloha" and body["event"] == "convert"
    assert body["preset"] == "hat" and body["fabric"] == "tee" and body["colors"] == 1
    assert body["stitches"] == r.json()["stats"]["stitches"] and body["formats"] == ["dst", "pes"]

    # A digitizing failure is reported as an error event.
    blank = png_bytes(np.zeros((50, 50), np.uint8))
    r = client.post("/api/digitize", files={"file": ("a.png", blank, "image/png")})
    assert r.status_code == 422 and sent[-1][2]["event"] == "error"


def test_stats_off_without_secret(client, monkeypatch):
    monkeypatch.setattr(main, "STATS_SECRET", "")
    monkeypatch.setattr(main.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")})
    assert r.status_code == 200 and r.json()["stats_recorded"] is False


def test_stats_failure_never_breaks_conversion(client, monkeypatch):
    monkeypatch.setattr(main, "STATS_SECRET", "s3cret")
    def boom(*a, **k): raise OSError("network down")
    monkeypatch.setattr(main.urllib.request, "urlopen", boom)
    r = client.post("/api/digitize", files={"file": ("a.png", text_png(), "image/png")})
    assert r.status_code == 200 and r.json()["stats_recorded"] is False
