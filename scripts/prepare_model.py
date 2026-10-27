"""Stage 5: package the evaluated .pt checkpoint for the FastAPI website."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.workflow import PROJECT_ROOT, configure_runtime, experiment_paths
from scripts.evaluate import training_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    return parser.parse_args(argv)


def prepare_model(paths):
    """Copy only a checkpoint whose evaluation matches this dataset."""
    checkpoint = training_checkpoint(paths)
    report_path = paths.evaluation / 'evaluation.json'
    manifest_path = paths.prepared / 'manifest.json'
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('Evaluasi atau manifest belum tersedia. Jalankan tahap sebelumnya.')
    if paths.deployment.exists() and any(paths.deployment.iterdir()):
        raise FileExistsError(f'Model deployment sudah ada: {paths.deployment}')

    training = json.loads((paths.root / 'training-report.json').read_text(encoding='utf-8'))
    evaluation = json.loads(report_path.read_text(encoding='utf-8'))
    with manifest_path.open('rb') as stream:
        manifest_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    digest = training['best_checkpoint_sha256']
    if evaluation['checkpoint_sha256'] != digest or evaluation['manifest_sha256'] != manifest_digest:
        raise ValueError('Hasil evaluasi tidak cocok dengan checkpoint atau dataset eksperimen ini')
    if tuple(evaluation['class_names']) != CLASS_NAMES:
        raise ValueError('Urutan kelas evaluasi tidak sesuai model PCB')
    confidence = float(evaluation['validation']['selected_confidence']['confidence'])
    if not 0.05 <= confidence <= 0.95:
        raise ValueError('Confidence dari validation tidak valid')

    metadata = {
        'name': 'YOLO11n PCB',
        'checkpoint_sha256': digest,
        'class_names': list(CLASS_NAMES),
        'default_confidence': confidence,
        'pretrained_source': 'Ultralytics yolo11n.pt / COCO',
        'training_method': 'full fine-tuning on PCB dataset',
        'manifest_sha256': manifest_digest,
        'input_size': 640,
        'nms_iou': 0.5,
        'training': training,
        'validation': evaluation['validation'],
        'test': evaluation['test'],
        'dataset_license': 'Unknown on Kaggle; checkpoint remains local',
    }
    paths.deployment.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, paths.deployment / 'pcb-best.pt')
    (paths.deployment / 'model-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return paths.deployment


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    folder = prepare_model(paths)
    print(f'Model .pt dan metadata siap untuk FastAPI di {folder}')


if __name__ == '__main__':
    main()
