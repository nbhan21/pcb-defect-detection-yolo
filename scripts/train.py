"""Train YOLO11n on the corrected Kaggle train/val split."""

import argparse
from pathlib import Path

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


DATASET_YAML = PROJECT_ROOT / 'data' / 'kaggle-corrected' / 'dataset.yaml'
PRETRAINED_WEIGHTS = PROJECT_ROOT / 'models' / 'base' / 'yolo11n.pt'


def train_model(paths, data: Path, weights: Path, *, epochs: int = 40, batch: int = 4,
                device: str = '0') -> Path:
    if not data.is_file() or not weights.is_file():
        raise FileNotFoundError('Dataset atau yolo11n.pt belum tersedia. Jalankan tahap persiapan data.')
    if paths.training.exists():
        raise FileExistsError(f'Hasil training sudah ada: {paths.training}. Pilih nama eksperimen lain.')

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.train(
        data=str(data), epochs=epochs, batch=batch, imgsz=640, device=device, workers=0,
        patience=8, plots=True, project=str(paths.root), name='training',
        exist_ok=False,
    )
    return paths.training / 'weights' / 'best.pt'


def main(argv=None):
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, help='Nama run, misalnya belajar-01')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', choices=('0', 'cpu'), default='0')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1:
        parser.error('epochs dan batch harus positif')
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    best = train_model(paths, DATASET_YAML, PRETRAINED_WEIGHTS,
                       epochs=args.epochs, batch=args.batch, device=args.device)
    print(f'Training selesai. Periksa {best}')


if __name__ == '__main__':
    main()
