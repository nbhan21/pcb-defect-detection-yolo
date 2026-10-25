# PCB Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the main implementation. Independent frontend and API work may be delegated in parallel under the frozen contract below. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune YOLO11n on PCB defects and deliver a tested local FastAPI image-upload application with PNG/CSV downloads and honest evaluation.

**Architecture:** Offline data preparation/training promotes an evaluated checkpoint to a shared inference service. FastAPI serves a static frontend from the same origin. Results stay in browser memory, with one inference slot and no history database.

**Tech Stack:** Python 3.12 on D:, project `.venv`, PyTorch CUDA wheels compatible with driver 537.70, Ultralytics YOLO11n, FastAPI, Pillow, HTML/CSS/JavaScript, pytest and Node test runner.

**Spec:** `docs/superpowers/specs/2026-10-25-pcb-inspection-design.md`, approved for implementation by Naufal on 25 October 2026.

## Global Constraints

- Work exclusively within this project; project writes, caches, temp, weights and outputs reside on D:.
- Retain raw data and ZIP. Repair 2,667 filename mismatches only in prepared labels.
- Preserve class order: mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper.
- Split by `light_##` source family with seed 42 and 6/2/2 families where duplicate analysis leaves 10 independent groups; linked duplicate families form indivisible supergroups and retain approximately 60/20/20 group proportions.
- Validation selects checkpoints and confidence; test is evaluated after selection and never drives tuning.
- One-image JPEG/PNG input, at most 10 MiB and 25 megapixels; no stored inspection history.
- Local single-worker FastAPI at 127.0.0.1:8000; missing or incompatible PCB weights produce 503, never COCO predictions passed off as PCB detections.
- Use industrial-brutalist blue/light palette from the spec, with no image filters or fabricated telemetry.
- Dataset, ZIP, caches, environment, checkpoints and source-derived images stay outside Git. Documentation is Markdown.

## Review Focus

1. A new upload or confidence change while a request finishes must not show an older result or permit stale downloads (frontend state tests).
2. Image extension and MIME can be misleading; decode actual bytes and enforce decoded pixel limits before inference (API tests).
3. Non-default EXIF orientation must produce coordinates and PNG at the same oriented dimensions (inference/API tests).
4. A corrupt/missing checkpoint or wrong class order must leave the UI usable with a clear not-ready status (service/API tests).
5. Source augmentations and duplicate images must not leak across the new split; label names must map bijectively (data preparation tests).

## Frozen Integration Contract

`GET /api/health`: `{ready, message, model:{name,checkpoint_sha256}, device, default_confidence}`; use HTTP 503 with this JSON if not ready. `model` can be null when unavailable.

`POST /api/inspect`: multipart field `image`, optional numeric `confidence`. Return:

```json
{
  "image": {"width": 600, "height": 600},
  "detections": [{"class_id": 0, "class_name": "mouse_bite", "confidence": 0.82, "bbox_xyxy": [10, 20, 40, 60]}],
  "counts": {"mouse_bite": 1, "spur": 0, "missing_hole": 0, "short": 0, "open_circuit": 0, "spurious_copper": 0},
  "total_detections": 1,
  "inference_ms": 42.1,
  "confidence_threshold": 0.25,
  "model": {"name": "YOLO11n PCB", "checkpoint_sha256": "actual-sha256", "device": "cuda:0"},
  "annotated_image": "data:image/png;base64,..."
}
```

This JSON is a schema illustration, not an actual model result. `bbox_xyxy` uses original oriented image pixels. Errors use `detail` with readable text, except standard FastAPI validation errors which the frontend must also format. Confidence range 0.05–0.95; provisional default 0.25 until validation selection. Explicit busy response is 429; input/decode errors 400, unsupported format 415, limits 413, bad parameters 422, not ready 503. Runtime config comes from local model metadata, with fallback default 0.25 before a trained model is available.

## Task 1: Reproducible Environment and Safe Data Preparation

**Files:** `scripts/project-env.ps1`, `setup.ps1`, `requirements.in`, `requirements.txt`, `.gitignore`, `pcb_inspector/data.py`, `scripts/prepare_data.py`, `tests/test_data.py`.

**Interfaces:** `prepare_dataset(raw_root: Path, output_root: Path, report_root: Path, seed: int = 42) -> dict`. Produces `data/prepared/dataset.yaml`, manifest JSON and audit report/PNG previews. Raw files remain unchanged; prepared images use hard links on D: (copy fallback) and labels get exact matching stems.

- [ ] Create `.gitignore` before initializing Git, connect the existing public repo, and commit only source/documentation/configuration.
- [ ] Write project-scoped environment configuration for TEMP/TMP/TMPDIR, UV/PIP/Torch/YOLO/matplotlib/XDG caches and outputs. Reuse the D: base Python and D: uv tool; create a new `.venv`.
- [ ] Verify available package versions, install a compatible single CUDA build (initial candidate torch 2.5.1 cu121 + torchvision 0.20.1), then freeze resolved versions after GPU verification. Do not install a CUDA toolkit or update drivers.
- [ ] Write failing tests for `_256`→`_600` bijection, invalid/ambiguous labels, normalized geometry, deterministic family splits, class coverage, and duplicate supergroups.
- [ ] Run `.venv/Scripts/python.exe -m pytest tests/test_data.py -q` and confirm RED before implementing data functions.
- [ ] Implement validated preparation. Record raw pair mapping, source family, image hash, output split and annotation counts. Check exact duplicate image bytes across families and merge connected groups before split. Fail clearly on unrecognized family or unresolved pair.
- [ ] Run tests to GREEN; run preparation on actual data. Visually inspect bounding-box previews from all classes plus original/rotated variants. Stop for diagnosis if coordinate geometry does not match images.
- [ ] Record actual new split counts and hashes; verify family/hash intersections between splits are empty. Commit source and aggregate reports, excluding source-image previews.

## Task 2: Fine-Tuning and Evaluation

**Files:** `scripts/train.py`, `scripts/evaluate.py`, `pcb_inspector/constants.py`, `tests/test_training_config.py`, local `models/` and `reports/`.

**Interfaces:** CLI training/evaluation scripts read the prepared YAML. Promotion produces `models/pcb-best.pt` and `models/model-metadata.json` with `name`, `checkpoint_sha256`, `class_names`, `default_confidence`, model provenance, split hash, configuration and actual validation/test metrics.

- [ ] Write failing configuration tests to prohibit test-as-validation and verify class order, provenance and split integrity; implement wrappers and run GREEN.
- [ ] Download trusted `yolo11n.pt` to D:. Verify torch version/CUDA/device and record driver/VRAM.
- [ ] Run one-epoch smoke training on a small representative train subset, with validation subset, image size 640 and candidate batch 4. On CUDA OOM retry batch 2 then 1, cleaning the failed model/process before retry. If GPU fails for other reasons, diagnose rather than silently starting a long CPU training.
- [ ] Baseline: selected smoke-test batch, image size 640, at most 40 epochs, early stopping patience 8, optimizer AdamW with lr0 0.001, cache false, workers 0, AMP when capability check passes, seed 42, all trainable layers. Set a one-hour trainer time budget (resource cap, not duration/performance promise). Use modest lighting augmentation, mosaic/mixup/cutmix disabled and no additional geometry that could cut tiny defects.
- [ ] Log training to D: and report actual progress while it runs. Select best checkpoint using validation.
- [ ] On validation only, choose confidence from 0.10–0.70 in increments of 0.05 by aggregate F1, preferring the higher recall candidate on ties, then the higher precision candidate. Keep NMS IoU 0.50 and record this choice.
- [ ] Run final held-out test once after selection; write per-class/aggregate precision, recall, mAP50/mAP50-95 and examples of failure types. Test visuals are analysis outputs, not tuning input.
- [ ] Copy selected weights to the promoted local path, compute hash and metadata. Do not publish data or weights while redistribution terms remain unclear.

## Task 3: API and Inference Service (Parallel Owner)

**Files:** `pcb_inspector/api.py`, `pcb_inspector/inference.py`, `pcb_inspector/constants.py`, `tests/test_api.py`, `tests/test_inference.py`, `run.ps1`.

**Interfaces:** `create_app(service=None) -> FastAPI`; `PCBDetector(model_path: Path, metadata_path: Path, device: str = 'auto')`, properties `ready`, `message`, `device`, `model_info`, `default_confidence`; `inspect(image: PIL.Image.Image, confidence: float) -> dict` returns the frozen inspection response. API owns its non-blocking inference lock. Constants module is owned by this task; training reads its class names without editing it.

- [ ] Write tests with injected lightweight predictor for health not-ready, valid image, invalid/empty/unsupported bytes, oversize payload and decoded pixel limit, confidence boundaries, zero detections, busy 429, model mapping mismatch and EXIF alignment.
- [ ] Run targeted tests RED, implement service/API, then run GREEN. Test fixtures are explicitly synthetic unit inputs and never application demo predictions.
- [ ] Load model once in app lifespan, use CUDA only if available and expose actual device; validate names against the canonical six classes. Apply EXIF transpose and RGB conversion before inference. Return sorted detections, class counts and readable PNG labels.
- [ ] Serve `web/` from the same FastAPI app, retain upload/result only for request lifetime, keep temporary files on D:, and return the documented errors without user paths/stack traces.
- [ ] Implement one-worker localhost launcher sourcing project environment. Leave weights absence as a visible not-ready state.

## Task 4: Industrial-Brutalist Frontend (Parallel Owner)

**Files:** `web/index.html`, `web/styles.css`, `web/app.js`, `web/helpers.mjs`, `tests/frontend.test.mjs`.

**Interfaces:** Consume only the frozen health/inspection endpoints. `helpers.mjs` exports CSV formatting and request-generation/state helpers for Node tests. No backend schema changes without coordinating with root/API owner.

- [ ] Write Node tests for CSV quoting/header-only result, error detail formatting, and stale request invalidation; run RED, implement helpers and run GREEN with `node --test tests/frontend.test.mjs`.
- [ ] Read project `.agents/skills/industrial-brutalist-ui/SKILL.md` and implement the approved light/blue palette. Keep PCB pixels unaffected, data readable, and keyboard controls accessible. No extra framework or Node build step.
- [ ] Build upload/drop zone, confidence control, original/result preview, actual model status, class totals, table and PNG/CSV downloads. Clear prior results on new file or confidence changes and disable stale downloads.
- [ ] Cover empty/loading/success/no-detection/error states, responsive single-column layout, and health 503 handling. Revoke blob URLs and ignore results from superseded requests.

## Task 5: Integration, Documentation and Publication

**Files:** `README.md`, `docs/learning-guide.md`, aggregate `reports/evaluation.json` and `reports/implementation-status.md`, relevant tests above.

- [ ] Integrate components using an actual trained checkpoint, verify readiness and a real image response. Never substitute fabricated detections for this gate.
- [ ] Run the full pytest suite and Node tests. Launch the local app and test upload-to-render through browser automation, actual PNG/CSV downloads, a bad file, zero-detection state, repeated file selection, and narrow layout.
- [ ] Inspect screenshot at desktop and narrow width; fix concrete usability/layout problems, then rerun the affected path only.
- [ ] Write README commands for setup, prepare, train, evaluate and run; add flow diagram, data/model attribution, actual metrics and limitations. Explain the frontend request, API validation, model output, rendering and downloads in `docs/learning-guide.md`.
- [ ] Verify staged files exclude raw data, ZIP, environment, caches, credentials, weights, and source-derived images. Commit and push source/docs to the already-authorized public repo.
- [ ] Update this checklist and actual status. Report what ran, measured performance, current limitations and how Naufal can run/inspect the demo.

## Execution Authorization

Naufal explicitly requested implementation after the system-design discussion. Preserve that authorization: proceed through the plan in this chat, using root-led execution and parallel independent UI/API tasks as previously permitted. Ask immediately only for a new user decision or a concrete scope change; do not add another routine approval checkpoint for this plan.
