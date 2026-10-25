"""Prepare immutable-source PCB data with family-disjoint evaluation splits."""
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import re
from zipfile import ZipFile


from .constants import CLASS_NAMES


def family_from_name(stem: str) -> str:
    match = re.search(r'(?:^|_)light_(\d{2})(?:_|$)', stem)
    if not match:
        raise ValueError(f'Family sumber tidak dikenal: {stem}')
    return f'light_{match.group(1)}'


def resolve_label(image: Path, labels_dir: Path) -> Path:
    candidates = [labels_dir / (image.stem + '.txt')]
    if image.stem.endswith('_600'):
        candidates.append(labels_dir / (image.stem[:-4] + '_256.txt'))
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f'Pasangan label hilang/ambigu: {image.name} ({len(found)} kandidat)')
    return found[0]


def parse_label(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError('Label YOLO harus berisi lima nilai per baris')
        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError('Class ID dan koordinat label tidak valid') from exc
        if class_id not in range(6) or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError('Class ID atau angka label di luar batas')
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError('Koordinat harus normalized, dengan luas positif')
        if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
            raise ValueError('Bounding box melewati batas gambar')
        boxes.append((class_id, x, y, w, h))
    return boxes


def split_families(records: list[dict], seed: int = 42) -> dict[str, str]:
    families = sorted({r['family'] for r in records})
    parents = {f: f for f in families}

    def find(f):
        while parents[f] != f:
            parents[f] = parents[parents[f]]
            f = parents[f]
        return f

    hashes = defaultdict(list)
    for record in records:
        hashes[record['sha256']].append(record['family'])
    for connected in hashes.values():
        for f in connected[1:]:
            a, b = find(connected[0]), find(f)
            parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for family in families:
        groups[find(family)].append(family)
    groups = sorted(groups.values())
    if len(groups) < 3:
        raise ValueError('Butuh minimal tiga grup independen untuk train/val/test')
    random.Random(seed).shuffle(groups)
    n_train = min(len(groups)-2, max(1, round(len(groups)*.6)))
    n_val = min(len(groups)-n_train-1, max(1, round(len(groups)*.2)))
    result = {}
    for index, group in enumerate(groups):
        split = 'train' if index < n_train else 'val' if index < n_train+n_val else 'test'
        for family in group:
            result[family] = split
    for split in ('train', 'val', 'test'):
        classes = {c for r in records if result[r['family']] == split for c in r['classes']}
        if classes != set(range(6)):
            raise ValueError(f'Split {split} tidak mencakup enam kelas: {sorted(classes)}')
    return result


def _preview(records, source_reader, destination: Path):
    from PIL import Image, ImageDraw, ImageFont
    selected = []
    for class_id in range(6):
        selected.append(next(r for r in records if class_id in r['classes'] and not r['stem'].startswith(('rotation_', 'l_'))))
    canvas = Image.new('RGB', (1260, 900), '#f4f7fc')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    for index, record in enumerate(selected):
        with Image.open(BytesIO(source_reader(record['image_path']))) as source:
            image = source.convert('RGB')
        boxes = parse_label(source_reader(record['label_path']).decode('utf-8-sig'))
        overlay = ImageDraw.Draw(image)
        w, h = image.size
        for c, x, y, bw, bh in boxes:
            rect = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
            overlay.rectangle(rect, outline='#ff3131', width=3)
            overlay.text((max(0,rect[0]),max(0,rect[1]-19)), CLASS_NAMES[c], font=font, fill='#ff3131', stroke_width=1)
        image.thumbnail((400, 400))
        left, top = (index % 3)*420+10, (index//3)*450+32
        canvas.paste(image, (left, top))
        draw.text((left, top-25), f'ANNOTATION / {CLASS_NAMES[index]}', font=font, fill='#111827')
        draw.text((left, top+405), record['stem'][:42], font=font, fill='#111827')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def prepare_dataset(raw_root: Path, output_root: Path, report_root: Path, seed: int = 42) -> dict:
    raw_root, output_root, report_root = map(lambda p: Path(p).resolve(), (raw_root, output_root, report_root))
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError('Data turunan harus terpisah dari folder raw')
    if (output_root/'manifest.json').exists():
        raise FileExistsError('Data prepared sudah ada; periksa manifest sebelum menyiapkan ulang')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Folder output tidak kosong')
    records, seen_labels, renamed = [], set(), 0
    archive_path = raw_root.parent/'pcb-defect-dataset.zip'
    archive = ZipFile(archive_path) if archive_path.exists() else None

    def read_source(path_string):
        path = Path(path_string)
        if archive:
            return archive.read(raw_root.name+'/'+path.relative_to(raw_root).as_posix())
        return path.read_bytes()

    try:
        for original_split in ('train', 'val', 'valid', 'test'):
            images_dir = raw_root/original_split/'images'
            if not images_dir.is_dir():
                continue
            labels_dir = raw_root/original_split/'labels'
            label_paths = {p.stem: p for p in labels_dir.glob('*.txt')}
            image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
            for image in image_paths:
                candidates = [image.stem]
                if image.stem.endswith('_600'):
                    candidates.append(image.stem[:-4]+'_256')
                found = [label_paths[c] for c in candidates if c in label_paths]
                if len(found) != 1:
                    raise ValueError(f'Pasangan label hilang/ambigu: {image.name}')
                label = found[0]
                if label in seen_labels:
                    raise ValueError(f'Label digunakan lebih dari sekali: {label.name}')
                seen_labels.add(label)
                boxes = parse_label(read_source(str(label)).decode('utf-8-sig'))
                renamed += label.stem != image.stem
                records.append({'stem': image.stem, 'filename': image.name, 'image_path': str(image), 'label_path': str(label), 'family': family_from_name(image.stem), 'sha256': hashlib.sha256(read_source(str(image))).hexdigest(), 'classes': sorted({b[0] for b in boxes}), 'boxes': len(boxes), 'original_split': original_split})
            if any(p not in seen_labels for p in label_paths.values()):
                raise ValueError(f'Label tanpa gambar di {original_split}')
            print(f'Audit {original_split}: {len(image_paths)} images', flush=True)
        if not records:
            raise ValueError('Tidak menemukan gambar dataset')
        split_map = split_families(records, seed)
        for split in ('train','val','test'):
            (output_root/split/'images').mkdir(parents=True, exist_ok=True)
            (output_root/split/'labels').mkdir(parents=True, exist_ok=True)
        names_in_split, link_counts = set(), Counter()
        for index, record in enumerate(records):
            split = split_map[record['family']]
            key = (split, record['stem'])
            if key in names_in_split:
                raise ValueError(f'Duplicate output filename: {key}')
            names_in_split.add(key)
            destination = output_root/split/'images'/record['filename']
            # Use the exact source audited above, including when the archive
            # and its extracted directory differ. Copies also isolate raw data.
            destination.write_bytes(read_source(record['image_path']))
            link_counts['copy'] += 1
            (output_root/split/'labels'/(record['stem']+'.txt')).write_bytes(read_source(record['label_path']))
            record['split'] = split
            if (index+1) % 2000 == 0:
                print(f'Prepared {index+1}/{len(records)} images', flush=True)
        manifest = {'seed': seed, 'family_split': split_map, 'records': records}
        encoded = json.dumps(manifest, indent=2).encode('utf-8')
        digest = hashlib.sha256(encoded).hexdigest()
        (output_root/'manifest.json').write_bytes(encoded)
        yaml_text = f'path: {output_root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n'
        yaml_text += ''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASS_NAMES))
        (output_root/'dataset.yaml').write_text(yaml_text, encoding='utf-8')
        summary = {'images': len(records), 'boxes': sum(r['boxes'] for r in records), 'renamed_labels': renamed, 'seed': seed, 'manifest_sha256': digest, 'family_split': split_map, 'split_counts': dict(Counter(r['split'] for r in records)), 'link_counts': dict(link_counts), 'class_names': list(CLASS_NAMES), 'original_split_not_used_for_evaluation': True}
        hashes_by_split = {s: {r['sha256'] for r in records if r['split']==s} for s in ('train','val','test')}
        if any(hashes_by_split[a] & hashes_by_split[b] for a,b in (('train','val'),('train','test'),('val','test'))):
            raise ValueError('Duplikasi image masih melewati split')
        summary['cross_split_duplicate_hashes'] = 0
        report_root.mkdir(parents=True, exist_ok=True)
        (report_root/'data-audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        _preview(records, read_source, report_root/'visuals'/'annotation-preview.png')
        return summary
    finally:
        if archive:
            archive.close()
