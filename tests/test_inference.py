"""Service tests isolate model loading so no GPU or external weights are used."""

import base64
import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from pcb_inspector.constants import CLASS_NAMES


def model_files(tmp_path, **changes):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"explicit-unit-test-model-fixture")
    metadata = {
        "name": "YOLO11n PCB",
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "class_names": list(CLASS_NAMES),
        "default_confidence": 0.35,
    }
    metadata.update(changes)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return weights, path


class TensorFixture:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class BoxesFixture:
    xyxy = TensorFixture([[2, 2, 12, 10], [15, 2, 22, 10]])
    cls = TensorFixture([0, 3])
    conf = TensorFixture([0.6, 0.9])


class PredictionFixture:
    boxes = BoxesFixture()


class ModelFixture:
    names = dict(enumerate(CLASS_NAMES))

    def predict(self, **kwargs):
        assert kwargs["conf"] == 0.35
        assert kwargs["device"] == "cpu"
        assert kwargs["iou"] == 0.5
        assert kwargs["imgsz"] == 640
        return [PredictionFixture()]


def test_missing_weights_does_not_try_loading_a_model(tmp_path, monkeypatch):
    import pcb_inspector.inference as module

    monkeypatch.setattr(module, "_load_yolo", lambda path: pytest.fail("must not download or load"))
    detector = module.PCBDetector(tmp_path / "absent.pt", tmp_path / "absent.json")
    assert detector.ready is False
    assert detector.model_info is None


@pytest.mark.parametrize("change", [
    {"checkpoint_sha256": "b" * 64},
    {"class_names": list(reversed(CLASS_NAMES))},
    {"default_confidence": float("nan")},
    {"default_confidence": 0.99},
])
def test_bad_metadata_never_loads_model(tmp_path, monkeypatch, change):
    import pcb_inspector.inference as module

    monkeypatch.setattr(module, "_load_yolo", lambda path: pytest.fail("metadata must be validated first"))
    detector = module.PCBDetector(*model_files(tmp_path, **change))
    assert detector.ready is False
    assert detector.model_info is None


def test_wrong_loaded_class_order_is_not_ready(tmp_path, monkeypatch):
    import pcb_inspector.inference as module

    class WrongModel(ModelFixture):
        names = dict(enumerate(reversed(CLASS_NAMES)))

    monkeypatch.setattr(module, "_load_yolo", lambda path: WrongModel())
    monkeypatch.setattr(module, "_choose_device", lambda device: "cpu")
    detector = module.PCBDetector(*model_files(tmp_path))
    assert detector.ready is False


def test_corrupt_checkpoint_is_safe_not_ready(tmp_path, monkeypatch):
    import pcb_inspector.inference as module

    def fail(path):
        raise ValueError("invalid serialized model")

    monkeypatch.setattr(module, "_load_yolo", fail)
    monkeypatch.setattr(module, "_choose_device", lambda device: "cpu")
    detector = module.PCBDetector(*model_files(tmp_path))
    assert detector.ready is False
    with pytest.raises(RuntimeError, match="belum siap"):
        detector.inspect(Image.new("RGB", (24, 12)), 0.35)


def test_predictions_are_sorted_counted_and_encoded_without_altering_dimensions(tmp_path, monkeypatch):
    import pcb_inspector.inference as module

    calls = []
    monkeypatch.setattr(module, "_load_yolo", lambda path: calls.append(path) or ModelFixture())
    monkeypatch.setattr(module, "_choose_device", lambda device: "cpu")
    detector = module.PCBDetector(*model_files(tmp_path))
    assert detector.ready is True
    result = detector.inspect(Image.new("RGB", (24, 12), "white"), 0.35)
    assert len(calls) == 1
    assert result["total_detections"] == 2
    assert result["counts"]["mouse_bite"] == 1
    assert result["counts"]["short"] == 1
    assert result["detections"][0]["class_name"] == "short"
    assert result["detections"][0]["bbox_xyxy"] == [15.0, 2.0, 22.0, 10.0]
    assert result["inference_ms"] >= 0
    assert result["model"]["checkpoint_sha256"] == detector.model_info["checkpoint_sha256"]
    with Image.open(BytesIO(base64.b64decode(result["annotated_image"].split(",", 1)[1]))) as png:
        assert png.size == (24, 12)
        assert png.format == "PNG"


def test_zero_boxes_returns_all_zero_counts(tmp_path, monkeypatch):
    import pcb_inspector.inference as module

    class EmptyPrediction:
        boxes = None

    class EmptyModel(ModelFixture):
        def predict(self, **kwargs):
            return [EmptyPrediction()]

    monkeypatch.setattr(module, "_load_yolo", lambda path: EmptyModel())
    monkeypatch.setattr(module, "_choose_device", lambda device: "cpu")
    detector = module.PCBDetector(*model_files(tmp_path))
    result = detector.inspect(Image.new("RGB", (24, 12)), 0.35)
    assert result["detections"] == []
    assert result["total_detections"] == 0
    assert set(result["counts"].values()) == {0}
