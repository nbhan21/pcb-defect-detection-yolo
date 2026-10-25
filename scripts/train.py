"""Fine-tune YOLO11n; all parameters and measured hardware are recorded."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from pcb_inspector.training import TrainingBudget, verify_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--time-hours', type=float, default=1.0)
    parser.add_argument('--name', default='baseline')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.time_hours <= 0:
        parser.error('epochs, batch, dan time-hours harus positif')
    prepared = root/'data/prepared'
    manifest = json.loads((prepared/'manifest.json').read_text())
    verify_manifest(manifest)
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    if not torch.cuda.is_available():
        raise RuntimeError('GPU CUDA belum tersedia. Diagnosis environment sebelum training panjang.')
    weights = root/'models/base/yolo11n.pt'
    if not weights.exists():
        raise FileNotFoundError(f'Pretrained checkpoint belum tersedia: {weights}')
    data_path = prepared/'dataset.yaml'
    if args.smoke:
        import yaml
        smoke = root/'data/smoke'
        smoke.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(data_path.read_text())
        for split, maximum in (('train',144),('val',72)):
            chosen = []
            families = sorted({r['family'] for r in manifest['records'] if r['split']==split})
            buckets = {f: [r for r in manifest['records'] if r['split']==split and r['family']==f] for f in families}
            # Stable spread through each family's source filenames, not only the first defect class.
            for family in families:
                rows = sorted(buckets[family],key=lambda r:r['filename'])
                count = max(1,maximum//len(families))
                indices = [min(len(rows)-1,int(i*len(rows)/count)) for i in range(min(count,len(rows)))]
                chosen.extend(str(prepared/split/'images'/rows[i]['filename']) for i in indices)
            listing = smoke/(split+'.txt')
            listing.write_text('\n'.join(chosen)+'\n',encoding='utf-8')
            config[split] = str(listing)
        config.pop('test',None)
        data_path = smoke/'dataset.yaml'
        data_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    config = dict(data=str(data_path), epochs=1 if args.smoke else args.epochs, batch=args.batch, imgsz=640, device=0, workers=0, cache=False, seed=42, deterministic=True, optimizer='AdamW', lr0=.001, patience=8, freeze=0, amp=True, mosaic=0.0, mixup=0.0, cutmix=0.0, degrees=0.0, translate=.0, scale=.0, fliplr=0.0, flipud=0.0, hsv_h=.0, hsv_s=.1, hsv_v=.1, project=str(root/'runs'), name='smoke' if args.smoke else args.name, exist_ok=False, save=True, plots=True, verbose=False)
    (root/'reports').mkdir(exist_ok=True)
    hardware = {'torch':torch.__version__, 'ultralytics':ultralytics_version, 'cuda_runtime':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0), 'vram_bytes':torch.cuda.get_device_properties(0).total_memory, 'config':config, 'training_from':'COCO pretrained yolo11n.pt', 'smoke':args.smoke}
    hardware['time_budget_hours'] = None if args.smoke else args.time_hours
    hardware['time_budget_note'] = 'Checked after each fit epoch; up to one epoch plus final validation may exceed the budget.'
    report = root/'reports'/('smoke-config.json' if args.smoke else 'training-config.json')
    report.write_text(json.dumps(hardware,indent=2),encoding='utf-8')
    print(json.dumps(hardware,indent=2),flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = YOLO(str(weights))
    budget = TrainingBudget(config['epochs'], None if args.smoke else args.time_hours*3600)
    model.add_callback('on_train_start', budget.start)
    model.add_callback('on_fit_epoch_end', budget.check)
    model.train(**config)
    hardware.update(elapsed_seconds=time.monotonic()-start, completed_epochs=model.trainer.epoch+1, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), best_checkpoint=str(model.trainer.best), save_dir=str(model.trainer.save_dir))
    hardware['best_checkpoint_sha256'] = hashlib.sha256(model.trainer.best.read_bytes()).hexdigest()
    report.write_text(json.dumps(hardware,indent=2),encoding='utf-8')
    print('TRAINING_FINISHED',json.dumps(hardware),flush=True)


if __name__ == '__main__':
    main()
