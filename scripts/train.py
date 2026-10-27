"""Stage 2: fine-tune pretrained YOLO11n on one prepared experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Nama folder eksperimen, misalnya belajar-01")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--time-hours", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.epochs < 1 or args.time_hours <= 0:
        parser.error("batch, epochs dan time-hours harus positif")
    return args


def check_training_inputs(paths):
    """Read the audited split and keep old training outputs untouched."""
    from pcb_inspector.training import verify_manifest

    manifest_path = paths.prepared / "manifest.json"
    dataset_yaml = paths.prepared / "dataset.yaml"
    weights = PROJECT_ROOT / "models" / "base" / "yolo11n.pt"
    for required in (manifest_path, dataset_yaml, weights):
        if not required.is_file():
            raise FileNotFoundError(f"Input training belum ada: {required}")
    if paths.training.exists():
        raise FileExistsError(f"Hasil training sudah ada: {paths.training}. Gunakan nama eksperimen baru.")
    verify_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    return weights


def training_options(paths, batch: int, epochs: int) -> dict:
    """Keep the YOLO parameters visible and editable in one place."""
    return {
        "data": str(paths.prepared / "dataset.yaml"),
        "epochs": epochs,
        "batch": batch,
        "imgsz": 640,
        "device": 0,
        "workers": 0,
        "cache": False,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 8,
        "freeze": 0,
        "amp": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.1,
        "hsv_v": 0.1,
        "project": str(paths.root),
        "name": "training",
        "exist_ok": False,
        "save": True,
        "plots": True,
        "verbose": False,
    }


def train_model(weights: Path, options: dict, time_hours: float) -> dict:
    """The Ultralytics training call and its measured outcome."""
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    from pcb_inspector.training import TrainingBudget

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA belum tersedia. Periksa environment sebelum training.")

    report = {
        "config": options,
        "training_from": "COCO pretrained yolo11n.pt",
        "time_budget_hours": time_hours,
        "time_budget_note": "Diperiksa pada akhir epoch; final validation dapat melewati batas.",
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = YOLO(str(weights))
    budget = TrainingBudget(options["epochs"], time_hours * 3600)
    model.add_callback("on_train_start", budget.start)
    model.add_callback("on_fit_epoch_end", budget.check)
    model.train(**options)

    best = Path(model.trainer.best)
    report.update(
        elapsed_seconds=time.monotonic() - started,
        completed_epochs=model.trainer.epoch + 1,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        best_checkpoint=str(best),
        best_checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
        save_dir=str(model.trainer.save_dir),
    )
    return report


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    weights = check_training_inputs(paths)
    options = training_options(paths, args.batch, args.epochs)
    report_path = paths.root / "training-report.json"
    if report_path.exists():
        raise FileExistsError(f"Laporan training sudah ada: {report_path}")
    report = train_model(weights, options, args.time_hours)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training selesai. Periksa {paths.training} dan {report_path}")


if __name__ == "__main__":
    main()
