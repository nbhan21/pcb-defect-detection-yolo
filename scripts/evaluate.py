"""Stage 3: select confidence on validation, then evaluate held-out test."""
import argparse
import hashlib
import json
from pathlib import Path

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.data import parse_label
from pcb_inspector.training import match_detections, select_confidence, verify_manifest
from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def training_checkpoint(paths) -> Path:
    """Use only the best checkpoint recorded by this experiment's training."""
    report_path = paths.root / 'training-report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Laporan training belum tersedia: {report_path}')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    checkpoint = Path(report['best_checkpoint'])
    if not checkpoint.is_file() or not checkpoint.resolve().is_relative_to(paths.training.resolve()):
        raise FileNotFoundError(f'Checkpoint eksperimen tidak ditemukan: {checkpoint}')
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != report['best_checkpoint_sha256']:
        raise ValueError('Checkpoint berubah sejak training; jalankan evaluasi dengan checkpoint yang benar')
    return checkpoint


def score(rows, threshold, class_id=None):
    """Count class-aware matches at one fixed confidence threshold."""
    tp = fp = fn = 0
    for row in rows:
        predictions = [
            item for item in row['predictions']
            if item['confidence'] >= threshold
            and (class_id is None or item['class_id'] == class_id)
        ]
        truths = [
            item for item in row['truths']
            if class_id is None or item['class_id'] == class_id
        ]
        matched, extra, missed = match_detections(predictions, truths)
        tp += matched
        fp += extra
        fn += missed
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'confidence': threshold, 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
    }


def predict_rows(model, prepared, split, cache_path, batch):
    """Cache predictions and ground truth for one evaluation split."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding='utf-8'))
    rows = []
    stream = model.predict(
        source=str(prepared / split / 'images'), stream=True, batch=batch,
        imgsz=640, conf=.05, iou=.50, device=0, verbose=False, max_det=300,
    )
    for index, result in enumerate(stream):
        height, width = result.orig_shape
        stem = Path(result.path).stem
        label_path = prepared / split / 'labels' / f'{stem}.txt'
        truths = []
        for class_id, x, y, box_width, box_height in parse_label(label_path.read_text(encoding='utf-8-sig')):
            truths.append({
                'class_id': class_id,
                'bbox_xyxy': [
                    (x - box_width / 2) * width, (y - box_height / 2) * height,
                    (x + box_width / 2) * width, (y + box_height / 2) * height,
                ],
            })
        predictions = []
        if result.boxes is not None:
            for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
                predictions.append({
                    'class_id': int(class_id), 'confidence': confidence,
                    'bbox_xyxy': [x1, y1, x2, y2],
                })
        rows.append({
            'filename': Path(result.path).name, 'width': width, 'height': height,
            'truths': truths, 'predictions': predictions,
        })
        if (index + 1) % 250 == 0:
            print(f'Predict {split}: {index + 1} images', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows), encoding='utf-8')
    return rows


def ap_summary(result):
    """Extract Ultralytics mAP values separately from fixed-threshold scores."""
    classes = {}
    for row, class_id in enumerate(result.box.ap_class_index.tolist()):
        classes[CLASS_NAMES[int(class_id)]] = {
            'mAP50': float(result.box.all_ap[row, 0]),
            'mAP50_95': float(result.box.all_ap[row].mean()),
        }
    return {
        'mAP50': float(result.box.map50),
        'mAP50_95': float(result.box.map),
        'per_class': classes,
    }


def rank_examples(rows, threshold):
    """Find six images with most errors and six with most true detections."""
    ranked = []
    for row in rows:
        predictions = [item for item in row['predictions'] if item['confidence'] >= threshold]
        tp, fp, fn = match_detections(predictions, row['truths'])
        ranked.append({
            'row': row, 'predictions': predictions,
            'tp': tp, 'fp': fp, 'fn': fn, 'errors': fp + fn,
        })
    errors = sorted(ranked, key=lambda item: (item['errors'], item['fn']), reverse=True)[:6]
    detections = sorted(ranked, key=lambda item: (item['tp'], -item['errors']), reverse=True)[:6]
    return errors, detections


def draw_examples(prepared, selection, destination):
    """Draw orange ground truth and blue predictions on a contact sheet."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=17)
    canvas = Image.new('RGB', (1260, 1440), '#f4f7fc')
    for index, item in enumerate(selection):
        row = item['row']
        with Image.open(prepared / 'test' / 'images' / row['filename']) as source_image:
            image = source_image.convert('RGB')
        draw = ImageDraw.Draw(image)
        for truth in row['truths']:
            draw.rectangle(truth['bbox_xyxy'], outline='#c2410c', width=2)
        for detection in item['predictions']:
            box = detection['bbox_xyxy']
            draw.rectangle(box, outline='#1d4ed8', width=3)
            label = f'{CLASS_NAMES[detection["class_id"]]} {detection["confidence"]:.2f}'
            draw.text(
                (box[0], max(0, box[1] - 20)), label, font=font,
                fill='#1d4ed8', stroke_width=1, stroke_fill='white',
            )
        image.thumbnail((400, 400))
        x, y = (index % 3) * 420 + 10, (index // 3) * 720 + 58
        canvas.paste(image, (x, y))
        caption = ImageDraw.Draw(canvas)
        caption.text(
            (x, y - 43), f'TEST / TP {item["tp"]} FP {item["fp"]} FN {item["fn"]}',
            font=font, fill='#111827',
        )
        caption.text((x, y + 410), row['filename'][:39], font=font, fill='#111827')
        caption.text((x, y + 438), 'BLUE prediction / ORANGE ground truth', font=font, fill='#111827')
    canvas.save(destination)


def save_examples(prepared, rows, threshold, directory):
    """Save small galleries and return the highest-error filenames."""
    directory.mkdir(parents=True, exist_ok=True)
    errors, detections = rank_examples(rows, threshold)
    draw_examples(prepared, errors, directory / 'test-errors.png')
    draw_examples(prepared, detections, directory / 'test-detections.png')
    return [
        {'filename': item['row']['filename'], 'tp': item['tp'], 'fp': item['fp'], 'fn': item['fn']}
        for item in errors
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error('batch harus positif')
    return args


def load_experiment(paths):
    """Check the prepared split and checkpoint before importing YOLO."""
    checkpoint = training_checkpoint(paths)
    manifest_path = paths.prepared / 'manifest.json'
    if not manifest_path.is_file() or not (paths.prepared / 'dataset.yaml').is_file():
        raise FileNotFoundError(f'Data prepared belum lengkap: {paths.prepared}')
    verify_manifest(json.loads(manifest_path.read_text(encoding='utf-8')))
    with manifest_path.open('rb') as stream:
        manifest_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with checkpoint.open('rb') as stream:
        checkpoint_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return checkpoint, checkpoint_digest, manifest_digest


def evaluate_model(model, paths, batch, checkpoint_digest, manifest_digest):
    """Freeze validation choices before looking at the test split."""
    cache = paths.evaluation / 'cache' / f'{checkpoint_digest[:12]}-{manifest_digest[:12]}'
    val_rows = predict_rows(model, paths.prepared, 'val', cache / 'val.json', batch)
    curve = [score(val_rows, round(.10 + i * .05, 2)) for i in range(13)]
    selected = select_confidence(curve, split='val')
    threshold = selected['confidence']
    print('Confidence dipilih dari validation:', threshold, flush=True)

    common = dict(
        data=str(paths.prepared / 'dataset.yaml'), batch=batch, imgsz=640,
        device=0, workers=0, conf=.001, iou=.50, plots=True,
        project=str(paths.evaluation), exist_ok=True, verbose=False,
    )
    validation_ap = ap_summary(model.val(split='val', name='validation', **common))

    # Test is used only after the checkpoint and threshold were fixed above.
    test_ap = ap_summary(model.val(split='test', name='test', **common))
    test_rows = predict_rows(model, paths.prepared, 'test', cache / 'test.json', batch)
    fixed = score(test_rows, threshold)
    per_class = {
        name: {**score(test_rows, threshold, i), **test_ap['per_class'].get(name, {})}
        for i, name in enumerate(CLASS_NAMES)
    }
    error_examples = save_examples(paths.prepared, test_rows, threshold, paths.evaluation / 'visuals')
    return {
        'checkpoint_sha256': checkpoint_digest,
        'manifest_sha256': manifest_digest,
        'class_names': list(CLASS_NAMES),
        'validation': {'selected_confidence': selected, 'confidence_candidates': curve, **validation_ap},
        'test': {**fixed, **test_ap, 'per_class': per_class, 'images': len(test_rows)},
        'error_examples': error_examples,
        'metric_notes': 'Fixed-threshold P/R/F1 use class-aware IoU >= 0.5. AP uses Ultralytics score curves. Confidence is chosen only on validation.',
    }


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    report_path = paths.evaluation / 'evaluation.json'
    if report_path.exists():
        raise FileExistsError(f'Evaluasi sudah ada: {report_path}')
    checkpoint, checkpoint_digest, manifest_digest = load_experiment(paths)

    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if len(model.names) != len(CLASS_NAMES) or tuple(model.names[i] for i in range(6)) != CLASS_NAMES:
        raise ValueError('Mapping kelas checkpoint tidak sesuai dataset PCB')
    report = evaluate_model(model, paths, args.batch, checkpoint_digest, manifest_digest)
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Evaluasi selesai. Periksa {report_path}')


if __name__ == '__main__':
    main()
