"""Predict defects in one PCB image using the trained best.pt."""

import argparse
import json
from pathlib import Path

from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths


def predict_image(paths, checkpoint: Path, image: Path, *, confidence: float = 0.25) -> tuple[Path, Path]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint tidak ditemukan: {checkpoint}')
    if not image.is_file():
        raise FileNotFoundError(f'Gambar tidak ditemukan: {image}')
    if not 0 < confidence <= 1:
        raise ValueError('Confidence harus lebih dari 0 dan paling tinggi 1')

    png_path = paths.predictions / f'{image.stem}-prediction.png'
    json_path = paths.predictions / f'{image.stem}-prediction.json'
    if png_path.exists() or json_path.exists():
        raise FileExistsError(f'Hasil gambar ini sudah ada: {png_path}')

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    result = model.predict(source=str(image), conf=confidence, imgsz=640, verbose=False)[0]

    paths.predictions.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(png_path))
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            class_id = int(class_id)
            detections.append({
                'class_id': class_id,
                'class_name': model.names[class_id],
                'confidence': score,
                'bbox_xyxy': box,
            })
    report = {'source_image': str(image.resolve()), 'checkpoint': str(checkpoint),
              'confidence_threshold': confidence, 'detections': detections}
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return png_path, json_path


def main(argv=None):
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    best = paths.training / 'weights' / 'best.pt'
    png, report = predict_image(paths, best, args.image, confidence=args.conf)
    print(f'Prediksi selesai: {png} dan {report}')


if __name__ == '__main__':
    main()
