"""Stage 4: run YOLO on one image and inspect its PNG and JSON result."""

import argparse
import json
from pathlib import Path

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths
from scripts.evaluate import training_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--image', type=Path, required=True, help='Satu gambar PCB JPG atau PNG di D:')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda:0'), default='auto')
    return parser.parse_args(argv)


def selected_confidence(paths) -> float:
    """Reuse the threshold chosen on validation, never pick one from test."""
    report_path = paths.evaluation / 'evaluation.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Evaluasi belum tersedia: {report_path}')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    threshold = float(report['validation']['selected_confidence']['confidence'])
    if not 0.05 <= threshold <= 0.95:
        raise ValueError('Confidence dari validation tidak valid')
    return threshold


def predict_image(paths, image: Path, requested_device: str) -> tuple[Path, Path]:
    """Keep the familiar Ultralytics predict call visible in this stage."""
    checkpoint = training_checkpoint(paths)
    confidence = selected_confidence(paths)
    if not image.is_file():
        raise FileNotFoundError(f'Gambar belum tersedia: {image}')
    png_path = paths.predictions / f'{image.stem}-prediction.png'
    json_path = paths.predictions / f'{image.stem}-prediction.json'
    if png_path.exists() or json_path.exists():
        raise FileExistsError('Prediksi untuk nama gambar ini sudah ada; gunakan gambar atau eksperimen lain.')

    import torch
    from ultralytics import YOLO

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise RuntimeError('CUDA belum tersedia; gunakan --device cpu')
    model = YOLO(str(checkpoint))
    if len(model.names) != len(CLASS_NAMES) or tuple(model.names[i] for i in range(6)) != CLASS_NAMES:
        raise ValueError('Mapping kelas checkpoint tidak sesuai dataset PCB')
    results = model.predict(
        source=str(image), conf=confidence, iou=0.5, imgsz=640,
        device=device, verbose=False, max_det=300,
    )
    result = results[0]
    paths.predictions.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(png_path))
    detections = []
    if result.boxes is not None:
        for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist():
            class_id = int(class_id)
            detections.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'confidence': score,
                'bbox_xyxy': [x1, y1, x2, y2],
            })
    report = {
        'source_image': str(image.resolve()),
        'checkpoint': str(checkpoint),
        'device': device,
        'confidence_threshold': confidence,
        'detections': detections,
        'annotated_image': str(png_path),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return png_path, json_path


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    png_path, json_path = predict_image(paths, args.image, args.device)
    print(f'Prediksi selesai. Periksa {png_path} dan {json_path}')


if __name__ == '__main__':
    main()
