import os
import random
import shutil
import time

import numpy as np
import torch

from src.config import NUM_CLASSES, IGNORE_INDEX, CLASS_PIXEL_FREQ, SEED


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


class Timer:
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._t0

    def start(self):
        self._t0 = time.time()

    def stop(self):
        self.elapsed = time.time() - self._t0
        return self.elapsed

# Checkpoint helpers 
def save_checkpoint(state, local_path, drive_path=None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save(state, local_path)
    if drive_path:
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        shutil.copy(local_path, drive_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model weights from path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    epoch = ckpt.get('epoch', 0)
    best_miou = ckpt.get('best_miou', 0.0)
    print(f'✅ Loaded checkpoint: epoch={epoch}, best_miou={best_miou:.4f}')
    return epoch, best_miou


# Loss function factory
def get_loss_fn(use_weights=True, device=None, num_classes=NUM_CLASSES,
                 ignore_index=IGNORE_INDEX):
    """
    CrossEntropyLoss. Optionally uses inverse-frequency class weights
    computed from the approximate CamVid pixel-frequency table.
    """
    if device is None:
        device = get_device()

    weights = None
    if use_weights:
        freq = CLASS_PIXEL_FREQ
        w = 1.0 / (freq / freq.sum())
        weights = torch.FloatTensor(w / w.sum() * num_classes).to(device)
        print(f'   Using class weights: {weights.cpu().numpy().round(2)}')
    return torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
