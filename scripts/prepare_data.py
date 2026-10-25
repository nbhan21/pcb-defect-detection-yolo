"""Create prepared data; run from the project root via prepare-data.ps1."""
import argparse
import json
from pathlib import Path
from pcb_inspector.data import prepare_dataset

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=root/'data/raw/pcb-defect-dataset')
    parser.add_argument('--output', type=Path, default=root/'data/prepared')
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.raw, args.output, root/'reports'), indent=2))
