"""Stage 6: serve the chosen .pt model and the website with FastAPI."""

import argparse
import os
from pathlib import Path

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', help='Model hasil percobaan lu; tanpa ini memakai baseline lokal')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda:0'), default='auto')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port harus antara 1 dan 65535')
    return args


def model_files(project_root: Path, experiment: str | None) -> tuple[Path, Path]:
    folder = (
        experiment_paths(project_root, experiment).deployment
        if experiment else Path(project_root) / 'models'
    )
    return folder / 'pcb-best.pt', folder / 'model-metadata.json'


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    model_path, metadata_path = model_files(PROJECT_ROOT, args.experiment)
    if args.experiment and (not model_path.is_file() or not metadata_path.is_file()):
        raise FileNotFoundError('Model eksperimen belum disiapkan. Jalankan scripts.prepare_model.')
    os.environ['PCB_MODEL_PATH'] = str(model_path)
    os.environ['PCB_METADATA_PATH'] = str(metadata_path)
    os.environ['PCB_DEVICE'] = args.device

    import uvicorn
    uvicorn.run('pcb_inspector.api:app', host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
