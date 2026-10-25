"""Small evaluation primitives independent of the training framework."""
from collections import defaultdict
import time


class TrainingBudget:
    """Stop after the first epoch/time limit without rewriting YOLO's schedule.

    Checked after each fit epoch, so elapsed time can exceed the budget by one
    epoch and final validation. Framework early stopping remains effective.
    """

    def __init__(self, epochs, seconds=None, clock=time.monotonic):
        self.epochs, self.seconds, self.clock = epochs, seconds, clock
        self.started = None

    def start(self, trainer):
        self.started = self.clock()

    def check(self, trainer):
        elapsed = self.clock() - self.started
        trainer.stop |= trainer.epoch + 1 >= self.epochs or (
            self.seconds is not None and elapsed >= self.seconds
        )


def verify_manifest(manifest: dict) -> None:
    by_family, by_hash = defaultdict(set), defaultdict(set)
    splits = set()
    for row in manifest['records']:
        split = row['split']
        splits.add(split)
        by_family[row['family']].add(split)
        by_hash[row['sha256']].add(split)
    if splits != {'train','val','test'}:
        raise ValueError('Manifest harus memiliki train, val, test')
    if any(len(v) != 1 for v in by_family.values()) or any(len(v) != 1 for v in by_hash.values()):
        raise ValueError('Source family/hash bocor antar split')


def _iou(a, b):
    width = max(0, min(a[2],b[2])-max(a[0],b[0]))
    height = max(0, min(a[3],b[3])-max(a[1],b[1]))
    intersection = width*height
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1]) + max(0,b[2]-b[0])*max(0,b[3]-b[1]) - intersection
    return intersection/union if union else 0.0


def match_detections(predictions: list[dict], truths: list[dict], iou_threshold: float = .5) -> tuple[int,int,int]:
    available = set(range(len(truths)))
    tp = 0
    for prediction in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        candidates = [(i,_iou(prediction['bbox_xyxy'], truth['bbox_xyxy'])) for i,truth in enumerate(truths) if i in available and truth['class_id']==prediction['class_id']]
        if candidates:
            best, overlap = max(candidates, key=lambda pair: pair[1])
            if overlap >= iou_threshold:
                tp += 1
                available.remove(best)
    return tp, len(predictions)-tp, len(truths)-tp


def select_confidence(metrics: list[dict], split: str = 'val') -> dict:
    if split != 'val':
        raise ValueError('Confidence hanya dipilih menggunakan validation')
    if not metrics:
        raise ValueError('Tidak ada kandidat confidence')
    def rank(row):
        p,r = row['precision'],row['recall']
        f1 = 2*p*r/(p+r) if p+r else 0
        return (round(f1,12),r,p,row['confidence'])
    result = dict(max(metrics, key=rank))
    p,r = result['precision'],result['recall']
    result['f1'] = 2*p*r/(p+r) if p+r else 0
    return result
