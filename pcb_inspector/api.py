"""Single-origin local HTTP application for one-image PCB inspection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import threading
import warnings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .constants import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from .inference import PCBDetector, normalize_image

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decode_image(payload: bytes) -> Image.Image:
    if not payload:
        raise HTTPException(400, "File gambar kosong.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                if source.format not in ("JPEG", "PNG"):
                    raise HTTPException(415, "Gunakan gambar JPEG atau PNG.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Resolusi gambar melebihi batas 25 megapiksel.")
                source.load()
                return normalize_image(source)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "Resolusi gambar melebihi batas yang didukung.") from None
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, "Gambar tidak dapat dibaca. File mungkin rusak.") from None


def create_app(service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.service is None:
            application.state.service = await run_in_threadpool(
                PCBDetector,
                Path(os.environ.get("PCB_MODEL_PATH", PROJECT_ROOT / "models" / "pcb-best.pt")),
                Path(os.environ.get("PCB_METADATA_PATH", PROJECT_ROOT / "models" / "model-metadata.json")),
                os.environ.get("PCB_DEVICE", "auto"),
            )
        yield

    application = FastAPI(title="PCB Inspector", version="1.0.0", lifespan=lifespan)
    application.state.service = service
    application.state.inference_lock = threading.Lock()

    @application.get("/api/health")
    async def health():
        detector = application.state.service
        if detector is None:
            return JSONResponse({"ready": False, "message": "Model PCB belum siap.", "model": None, "device": "unavailable", "default_confidence": 0.25}, status_code=503)
        return JSONResponse({"ready": detector.ready, "message": detector.message, "model": detector.model_info, "device": detector.device, "default_confidence": detector.default_confidence}, status_code=200 if detector.ready else 503)

    @application.post("/api/inspect")
    async def inspect(image: UploadFile = File(...), confidence: str | None = Form(None)):
        detector = application.state.service
        if detector is None or not detector.ready:
            await image.close()
            raise HTTPException(503, "Model PCB belum siap. Periksa status model sebelum inspeksi.")
        try:
            selected_confidence = detector.default_confidence if confidence is None else float(confidence)
        except (ValueError, TypeError):
            await image.close()
            raise HTTPException(422, "Confidence harus berupa angka antara 0.05 dan 0.95.") from None
        if not math.isfinite(selected_confidence) or not 0.05 <= selected_confidence <= 0.95:
            await image.close()
            raise HTTPException(422, "Confidence harus berada antara 0.05 dan 0.95.")
        try:
            if image.content_type not in (None, "", "image/jpeg", "image/png", "application/octet-stream"):
                raise HTTPException(415, "Gunakan gambar JPEG atau PNG.")
            payload = await image.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await image.close()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Ukuran file melebihi batas 10 MiB.")
        if not application.state.inference_lock.acquire(blocking=False):
            raise HTTPException(429, "Inspeksi lain sedang berjalan. Coba kembali setelah selesai.")

        def process():
            decoded = _decode_image(payload)
            try:
                return detector.inspect(decoded, selected_confidence)
            finally:
                decoded.close()

        try:
            return await run_in_threadpool(process)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Inference PCB gagal")
            raise HTTPException(500, "Inspeksi gagal diproses. Silakan coba kembali.") from None
        finally:
            application.state.inference_lock.release()

    @application.get("/", include_in_schema=False)
    async def index():
        index_path = PROJECT_ROOT / "web" / "index.html"
        if not index_path.is_file():
            return JSONResponse({"detail": "Frontend belum tersedia."}, status_code=503)
        return FileResponse(index_path)

    application.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "web"), check_dir=False), name="web")
    return application


app = create_app()
