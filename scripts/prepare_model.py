"""Copy the trained .pt model into the FastAPI deployment folder."""

import argparse
import json
from pathlib import Path
import shutil

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def prepare_model(paths, checkpoint: Path) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint tidak ditemukan: {checkpoint}')
    if paths.deployment.exists() and any(paths.deployment.iterdir()):
        raise FileExistsError(f'Model deployment sudah ada: {paths.deployment}')

    paths.deployment.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, paths.deployment / 'pcb-best.pt')
    metadata = {
        'name': 'YOLO11n PCB',
        'class_names': list(CLASS_NAMES),
        'default_confidence': 0.25,
    }
    (paths.deployment / 'model-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return paths.deployment


def main(argv=None):
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    checkpoint = paths.training / 'weights' / 'best.pt'
    folder = prepare_model(paths, checkpoint)
    print(f'Model untuk FastAPI siap di {folder}')


if __name__ == '__main__':
    main()
