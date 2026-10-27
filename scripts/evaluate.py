"""Evaluate best.pt on Kaggle's validation and test splits with YOLO."""

import argparse
import json
from pathlib import Path

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths
from scripts.train import DATASET_YAML


def metrics_summary(metrics) -> dict:
    """Keep the reported values in Ultralytics' own metric definitions."""
    return {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
    }


def evaluate_model(paths, data: Path, checkpoint: Path, *, batch: int = 4,
                   device: str = '0') -> dict:
    if not data.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Dataset atau best.pt belum tersedia. Jalankan persiapan dan training.')
    report_path = paths.evaluation / 'evaluation.json'
    if report_path.exists():
        raise FileExistsError(f'Hasil evaluasi sudah ada: {report_path}')

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    report = {'dataset': str(data), 'checkpoint': str(checkpoint)}
    for split in ('val', 'test'):
        metrics = model.val(
            data=str(data), split=split, batch=batch, imgsz=640, device=device,
            workers=0, plots=True, project=str(paths.evaluation), name=split, exist_ok=False,
        )
        report['validation' if split == 'val' else 'test'] = metrics_summary(metrics)
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', choices=('0', 'cpu'), default='0')
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error('batch harus positif')
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    best = paths.training / 'weights' / 'best.pt'
    report = evaluate_model(paths, DATASET_YAML, best, batch=args.batch, device=args.device)
    print(json.dumps(report, indent=2))
    print(f'Plot dan metrik ada di {paths.evaluation}')


if __name__ == '__main__':
    main()
