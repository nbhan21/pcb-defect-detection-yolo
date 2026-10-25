"""Load a verified local PCB checkpoint and turn real detections into results."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import CLASS_NAMES, DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)


def normalize_image(image: Image.Image) -> Image.Image:
    """Use oriented RGB pixels, removing source metadata from result images."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.info.clear()
    return normalized


def _load_yolo(path: Path):
    # This import is deliberately delayed until trusted local weights exist.
    from ultralytics import YOLO

    return YOLO(str(path), task="detect")


def _choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested in ("cuda:0", "0") and torch.cuda.is_available():
        return "cuda:0"
    raise ValueError("Device inference tidak tersedia.")


def _ordered_names(names) -> tuple[str, ...]:
    if isinstance(names, dict):
        if set(names) != set(range(len(CLASS_NAMES))):
            raise ValueError("Indeks kelas checkpoint tidak cocok.")
        return tuple(names[index] for index in range(len(CLASS_NAMES)))
    return tuple(names)


def _png_data_uri(image: Image.Image, detections: list[dict]) -> str:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, min(24, round(min(canvas.size) / 40))))
    width = max(2, round(min(canvas.size) / 300))
    for detection in detections:
        box = detection["bbox_xyxy"]
        draw.rectangle(tuple(box), outline="#1D4ED8", width=width)
        label = f'{detection["class_name"]} {detection["confidence"]:.0%}'
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        x = min(max(0, round(box[0])), max(0, canvas.width - text_width - 8))
        y = min(max(0, round(box[1]) - text_height - 8), max(0, canvas.height - text_height - 8))
        draw.rectangle((x, y, x + text_width + 8, y + text_height + 8), fill="#DBEAFE", outline="#1D4ED8")
        draw.text((x + 4, y + 4 - text_box[1]), label, font=font, fill="#111827")
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class PCBDetector:
    """A readiness-aware detector; unavailable weights never trigger a download."""

    def __init__(self, model_path: Path, metadata_path: Path, device: str = "auto"):
        self.ready = False
        self.message = "Model PCB belum tersedia. Jalankan training dan evaluasi terlebih dahulu."
        self.device = "unavailable"
        self.model_info = None
        self.default_confidence = DEFAULT_CONFIDENCE
        self._model = None
        model_path, metadata_path = Path(model_path), Path(metadata_path)
        if not model_path.is_file() or not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            with model_path.open("rb") as weights:
                digest = hashlib.file_digest(weights, "sha256").hexdigest()
            if metadata.get("checkpoint_sha256") != digest:
                raise ValueError("Hash checkpoint berbeda dari metadata.")
            if tuple(metadata.get("class_names", ())) != CLASS_NAMES:
                raise ValueError("Mapping kelas metadata tidak cocok.")
            confidence = float(metadata.get("default_confidence", DEFAULT_CONFIDENCE))
            if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
                raise ValueError("Confidence metadata tidak valid.")
            selected_device = _choose_device(device)
            model = _load_yolo(model_path)
            if _ordered_names(model.names) != CLASS_NAMES:
                raise ValueError("Mapping kelas checkpoint tidak cocok.")
            self._model = model
            self.device = selected_device
            self.default_confidence = confidence
            self.model_info = {"name": str(metadata.get("name") or "YOLO11n PCB"), "checkpoint_sha256": digest}
            self.message = "Model PCB siap digunakan."
            self.ready = True
        except Exception:
            logger.exception("Checkpoint PCB tidak dapat dimuat")
            self.message = "Model PCB belum siap. Periksa checkpoint dan metadata lokal."

    def inspect(self, image: Image.Image, confidence: float) -> dict:
        if not self.ready or self._model is None:
            raise RuntimeError("Model PCB belum siap.")
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            raise ValueError("Confidence harus berada antara 0.05 dan 0.95.")
        oriented = normalize_image(image)
        started = time.perf_counter()
        results = self._model.predict(
            source=oriented,
            conf=confidence,
            iou=0.50,
            imgsz=640,
            device=self.device,
            verbose=False,
            max_det=300,
        )
        detections = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            for xyxy, class_value, score in zip(coordinates, classes, scores, strict=True):
                class_id = int(class_value)
                if class_id not in range(len(CLASS_NAMES)) or class_value != class_id:
                    raise ValueError("Indeks kelas prediksi tidak sesuai model PCB.")
                if len(xyxy) != 4 or not all(math.isfinite(float(value)) for value in (*xyxy, score)):
                    raise ValueError("Koordinat prediksi tidak valid.")
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                bounded = [max(0.0, min(oriented.width, x1)), max(0.0, min(oriented.height, y1)), max(0.0, min(oriented.width, x2)), max(0.0, min(oriented.height, y2))]
                if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                    continue
                detections.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "confidence": float(score), "bbox_xyxy": bounded})
        inference_ms = (time.perf_counter() - started) * 1000
        detections.sort(key=lambda entry: entry["confidence"], reverse=True)
        counts = {name: 0 for name in CLASS_NAMES}
        for detection in detections:
            counts[detection["class_name"]] += 1
        return {
            "image": {"width": oriented.width, "height": oriented.height},
            "detections": detections,
            "counts": counts,
            "total_detections": len(detections),
            "inference_ms": round(inference_ms, 3),
            "confidence_threshold": confidence,
            "model": {**self.model_info, "device": self.device},
            "annotated_image": _png_data_uri(oriented, detections),
        }
