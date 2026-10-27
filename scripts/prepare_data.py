"""Copy Kaggle's train/val/test split with matching image and label names."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime


RAW_DATA = PROJECT_ROOT / 'data' / 'raw' / 'pcb-defect-dataset'
CORRECTED_DATA = PROJECT_ROOT / 'data' / 'kaggle-corrected'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def prepare_dataset(raw_root: Path, output_root: Path) -> dict:
    """Audit every pair, then copy it without changing pixels or annotations."""
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f'Dataset raw tidak ditemukan: {raw_root}')
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError('Folder output harus terpisah dari dataset raw')
    if output_root.exists():
        raise FileExistsError(f'Dataset hasil sudah ada: {output_root}')

    pairs = []
    split_counts = {}
    renamed_labels = {}
    for split in ('train', 'val', 'test'):
        images_dir = raw_root / split / 'images'
        labels_dir = raw_root / split / 'labels'
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f'Folder images/labels tidak lengkap: {split}')
        images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = {p.stem: p for p in labels_dir.glob('*.txt')}
        if not images:
            raise ValueError(f'Tidak ada gambar pada split {split}')
        used_labels = set()
        output_names = set()
        renamed = 0
        for image in images:
            candidates = [image.stem]
            if image.stem.endswith('_600'):
                candidates.append(image.stem[:-4] + '_256')
            matches = [labels[name] for name in candidates if name in labels]
            if len(matches) != 1:
                raise ValueError(f'Pasangan label hilang atau ambigu: {image.name} ({split})')
            label = matches[0]
            if label in used_labels or image.stem in output_names:
                raise ValueError(f'Pasangan atau nama output berulang: {image.name} ({split})')
            used_labels.add(label)
            output_names.add(image.stem)
            renamed += label.stem != image.stem
            pairs.append((split, image, label))
        if len(used_labels) != len(labels):
            raise ValueError(f'Ada label tanpa gambar pada split {split}')
        split_counts[split] = len(images)
        renamed_labels[split] = renamed

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output_root.name}-', dir=output_root.parent) as temporary:
        stage = Path(temporary)
        for split, image, label in pairs:
            images_out = stage / split / 'images'
            labels_out = stage / split / 'labels'
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, images_out / image.name)
            shutil.copy2(label, labels_out / f'{image.stem}.txt')

        yaml_text = f'path: {output_root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n'
        yaml_text += ''.join(f'  {index}: {name}\n' for index, name in enumerate(CLASS_NAMES))
        (stage / 'dataset.yaml').write_text(yaml_text, encoding='utf-8')
        summary = {'images': len(pairs), 'split_counts': split_counts, 'renamed_labels': renamed_labels}
        (stage / 'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        stage.rename(output_root)
    return summary


def main(argv=None):
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW_DATA)
    parser.add_argument('--output', type=Path, default=CORRECTED_DATA)
    args = parser.parse_args(argv)
    summary = prepare_dataset(args.raw, args.output)
    print(json.dumps(summary, indent=2))
    print(f'Dataset siap: {args.output}')


if __name__ == '__main__':
    main()
