"""HTTP behavior tests use an injected detector, never demo predictions."""

import base64
from io import BytesIO
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pcb_inspector.constants import CLASS_NAMES


def picture(fmt="PNG", size=(24, 12), exif=None):
    stream = BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", size, "white").save(stream, format=fmt, **kwargs)
    return stream.getvalue()


class DetectorFixture:
    """Explicit controlled inference boundary for API-only unit tests."""

    ready = True
    message = "Model siap."
    device = "cpu"
    model_info = {"name": "unit-test-fixture", "checkpoint_sha256": "a" * 64}
    default_confidence = 0.35

    def inspect(self, image, confidence):
        stream = BytesIO()
        image.save(stream, "PNG")
        return {
            "image": {"width": image.width, "height": image.height},
            "detections": [],
            "counts": {name: 0 for name in CLASS_NAMES},
            "total_detections": 0,
            "inference_ms": 0.1,
            "confidence_threshold": confidence,
            "model": {**self.model_info, "device": self.device},
            "annotated_image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
        }


@pytest.fixture
def client():
    from pcb_inspector.api import create_app

    with TestClient(create_app(DetectorFixture())) as value:
        yield value


def test_missing_checkpoint_health_is_503_and_root_stays_accessible(tmp_path, monkeypatch):
    from pcb_inspector.api import create_app

    monkeypatch.setenv("PCB_MODEL_PATH", str(tmp_path / "absent.pt"))
    monkeypatch.setenv("PCB_METADATA_PATH", str(tmp_path / "absent.json"))
    with TestClient(create_app()) as local:
        result = local.get("/api/health")
        assert result.status_code == 503
        assert result.json()["ready"] is False
        assert result.json()["model"] is None
        assert local.get("/").status_code in (200, 503)
        assert local.post("/api/inspect", files={"image": ("x.png", picture(), "image/png")}).status_code == 503


def test_ready_health_has_metadata_and_default_threshold(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["default_confidence"] == 0.35
    assert response.json()["device"] == "cpu"


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_valid_image_yields_consistent_png_counts_and_metadata(client, fmt, mime):
    response = client.post("/api/inspect", files={"image": ("image", picture(fmt), mime)})
    assert response.status_code == 200
    result = response.json()
    assert result["image"] == {"width": 24, "height": 12}
    assert result["total_detections"] == 0
    assert list(result["counts"]) == list(CLASS_NAMES)
    assert result["confidence_threshold"] == 0.35
    png = base64.b64decode(result["annotated_image"].split(",", 1)[1])
    with Image.open(BytesIO(png)) as decoded:
        assert decoded.size == (24, 12)
        assert decoded.format == "PNG"


def test_actual_image_bytes_override_misleading_filename(client):
    response = client.post("/api/inspect", files={"image": ("not-a-photo.exe", picture(), "application/octet-stream")})
    assert response.status_code == 200


@pytest.mark.parametrize("payload", [b"", b"broken image bytes"])
def test_empty_or_corrupt_image_is_400(client, payload):
    response = client.post("/api/inspect", files={"image": ("photo.png", payload, "image/png")})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)


def test_unsupported_decoded_format_is_415(client):
    response = client.post("/api/inspect", files={"image": ("photo.png", picture("GIF"), "image/png")})
    assert response.status_code == 415


def test_plain_non_image_file_is_415(client):
    response = client.post("/api/inspect", files={"image": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_limit_is_enforced_before_decode(client):
    from pcb_inspector.constants import MAX_UPLOAD_BYTES

    response = client.post("/api/inspect", files={"image": ("large.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")})
    assert response.status_code == 413


def test_decoded_pixel_limit_is_enforced(client, monkeypatch):
    import pcb_inspector.api as module

    monkeypatch.setattr(module, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/api/inspect", files={"image": ("large.png", picture(size=(20, 20)), "image/png")})
    assert response.status_code == 413


@pytest.mark.parametrize("value", ["0.049", "0.951", "nan", "inf", "oops"])
def test_bad_confidence_is_422(client, value):
    response = client.post("/api/inspect", files={"image": ("x.png", picture(), "image/png")}, data={"confidence": value})
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["0.05", "0.95"])
def test_threshold_boundaries_are_accepted(client, value):
    response = client.post("/api/inspect", files={"image": ("x.png", picture(), "image/png")}, data={"confidence": value})
    assert response.status_code == 200
    assert response.json()["confidence_threshold"] == float(value)


def test_exif_orientation_matches_returned_image_dimensions(client):
    exif = Image.Exif()
    exif[274] = 6
    result = client.post("/api/inspect", files={"image": ("rotated.jpg", picture("JPEG", exif=exif), "image/jpeg")}).json()
    assert result["image"] == {"width": 12, "height": 24}
    with Image.open(BytesIO(base64.b64decode(result["annotated_image"].split(",", 1)[1]))) as decoded:
        assert decoded.size == (12, 24)
        assert not decoded.getexif()


def test_inference_error_is_readable_and_does_not_leak_path(client):
    def fail(image, confidence):
        raise RuntimeError(r"private D:\secret\weights.pt")

    client.app.state.service.inspect = fail
    response = client.post("/api/inspect", files={"image": ("x.png", picture(), "image/png")})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert client.get("/api/health").status_code == 200


def test_busy_request_returns_429_without_waiting_for_first_prediction():
    from pcb_inspector.api import create_app

    entered, release = threading.Event(), threading.Event()

    class BlockingDetector(DetectorFixture):
        def inspect(self, image, confidence):
            entered.set()
            assert release.wait(5)
            return super().inspect(image, confidence)

    with TestClient(create_app(BlockingDetector())) as local:
        responses = []
        worker = threading.Thread(target=lambda: responses.append(local.post("/api/inspect", files={"image": ("a.png", picture(), "image/png")})))
        worker.start()
        assert entered.wait(5)
        try:
            second = local.post("/api/inspect", files={"image": ("b.png", picture(), "image/png")})
            assert second.status_code == 429
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert responses[0].status_code == 200
