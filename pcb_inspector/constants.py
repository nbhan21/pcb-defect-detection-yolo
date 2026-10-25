"""Canonical class order used by data, training, evaluation and serving."""

CLASS_NAMES = (
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
)

DEFAULT_CONFIDENCE = 0.25
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
