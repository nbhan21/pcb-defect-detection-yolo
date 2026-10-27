"""Stage 1: audit existing YOLO labels and prepare a separate data split."""

import argparse
import json
from pathlib import Path

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Nama folder eksperimen, misalnya belajar-01")
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pcb-defect-dataset",
        help="Folder dataset Kaggle yang sudah diekstrak",
    )
    return parser.parse_args(argv)


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    from pcb_inspector.data import prepare_dataset

    summary = prepare_dataset(args.raw, paths.prepared, paths.root / "preparation")
    print(json.dumps(summary, indent=2))
    print(f"Periksa dataset.yaml, manifest.json dan preview di {paths.root}")


if __name__ == "__main__":
    main()
