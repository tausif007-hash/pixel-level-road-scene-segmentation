"""
metrics.py
Evaluation metrics: running-average meter, and a confusion-matrix-based
segmentation metrics accumulator 
"""

import numpy as np

from src.config import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES


class AverageMeter:
    """Tracks a running average of a scalar value ."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


class SegmentationMetrics:
    """
    this part accumulates a confusion matrix over an epoch.
    """

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.K = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion[:] = 0

    def update(self, preds, targets):
        preds = preds.detach().cpu().numpy().astype(np.int64).flatten()
        targets = targets.detach().cpu().numpy().astype(np.int64).flatten()
        valid = targets != self.ignore_index
        preds, targets = preds[valid], targets[valid]
        preds = np.clip(preds, 0, self.K - 1)
        indices = self.K * targets + preds
        cm = np.bincount(indices, minlength=self.K ** 2)
        self.confusion += cm.reshape(self.K, self.K)

    def compute(self):
        cm = self.confusion
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        pa = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
        iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
        present = cm.sum(axis=1) > 0
        miou = float(np.nanmean(iou[present])) if present.any() else 0.0
        return pa, miou, iou.tolist()

    def print_table(self):
        _, miou, per_class = self.compute()
        print(f"\n{'Class':<16} {'IoU':>8}  Bar")
        print('─' * 40)
        for name, iou in zip(CLASS_NAMES, per_class):
            if not np.isnan(iou):
                bar = '█' * int(iou * 20)
                print(f'{name:<16} {iou*100:>5.1f}%  {bar}')
            else:
                print(f'{name:<16}   N/A')
        print('─' * 40)
        print(f'{"mIoU":<16} {miou*100:>5.1f}%\n')
