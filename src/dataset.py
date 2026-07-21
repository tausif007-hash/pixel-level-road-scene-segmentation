"""
dataset.py
"""

import os
import random
import tarfile
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src.config import (
    CONFIG, MEAN, STD, IGNORE_INDEX, RAW_TO_TRAIN, SEED, CAMVID_URL,
)

# Dataset download 
def download_camvid(data_root, url=CAMVID_URL):

    os.makedirs(data_root, exist_ok=True)
    tgz_path = os.path.join(data_root, 'camvid.tgz')

    if not os.path.isfile(tgz_path):
        print('Downloading CamVid (~65 MB)...')
        urllib.request.urlretrieve(url, tgz_path)
        print('Extracting...')
        with tarfile.open(tgz_path) as t:
            t.extractall(data_root)
        print('CamVid downloaded and extracted!')
    else:
        print('CamVid archive already exists, skipping download.')

    return data_root



# Root / label-naming discovery
def find_camvid_root(base):
    for root, dirs, _files in os.walk(base):
        if 'images' in dirs and 'labels' in dirs:
            return root
    return None


def detect_label_name(img_fname, lbl_dir):
    stem = img_fname.replace('.png', '')
    candidates = [
        img_fname,
        f'{stem}_L.png',
        f'{stem}_P.png',
        f'{stem}_annot.png',
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(lbl_dir, candidate)):
            return candidate
    return None

# Train / val / test split builder
def build_splits(root, valid_txt='valid.txt', test_fraction=0.15, seed=SEED):
    img_dir = os.path.join(root, 'images')
    lbl_dir = os.path.join(root, 'labels')
    valid_path = os.path.join(root, valid_txt)

    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if len(all_imgs) == 0:
        raise FileNotFoundError(f'No PNG images found in {img_dir}')

    # Load val filenames from valid.txt
    val_names = set()
    if os.path.isfile(valid_path):
        with open(valid_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    val_names.add(os.path.basename(line))
        print(f'valid.txt loaded: {len(val_names)} val filenames')
    else:
        print(' valid.txt not found — using random 15% as val')

    # Pair images with labels
    val_pairs, other_pairs = [], []
    skipped = 0

    for fname in all_imgs:
        lbl_name = detect_label_name(fname, lbl_dir)
        if lbl_name is None:
            skipped += 1
            continue
        pair = (os.path.join(img_dir, fname), os.path.join(lbl_dir, lbl_name))
        if fname in val_names or lbl_name in val_names:
            val_pairs.append(pair)
        else:
            other_pairs.append(pair)

    if skipped:
        print(f'Skipped {skipped} images with no matching label')

    # Fallback: valid.txt matched nothing → random 15% as val
    if len(val_pairs) == 0:
        random.seed(seed)
        random.shuffle(other_pairs)
        n_val = max(1, int(len(other_pairs) * 0.15))
        val_pairs = other_pairs[:n_val]
        other_pairs = other_pairs[n_val:]
        print(f' No val pairs from valid.txt — used random {n_val} as val')

    # Split remaining into train + test
    random.seed(seed)
    random.shuffle(other_pairs)
    n_test = max(1, int(len(other_pairs) * test_fraction))
    test_pairs = other_pairs[:n_test]
    train_pairs = other_pairs[n_test:]

    return {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}


def verify_dataset(root):
    actual_root = find_camvid_root(root)
    if actual_root is None:
        raise FileNotFoundError(
            'Cannot find images/ + labels/ folders under: ' + root
        )
    print(f'CamVid root detected : {actual_root}')

    splits = build_splits(actual_root)
    print('\n Split summary:')
    for name, pairs in splits.items():
        print(f'    {name:<8}: {len(pairs)} pairs')

    img_ex, lbl_ex = splits['train'][0]
    assert os.path.isfile(img_ex), f'Image missing: {img_ex}'
    assert os.path.isfile(lbl_ex), f'Label missing: {lbl_ex}'
    print('\n  Pair check passed')

    return actual_root, splits

# Label decoding: raw grayscale index → training class ID
LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.int64)
for _raw_id, _train_id in RAW_TO_TRAIN.items():
    LABEL_LUT[_raw_id] = _train_id


def decode_mask(mask_path: str) -> np.ndarray:
    gray = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
    return LABEL_LUT[gray].astype(np.int64)


# PyTorch Dataset
class CamVidDataset(Dataset):
    

    def __init__(self, pairs, split='train', img_size=(360, 480), augment=True):
        self.samples = pairs
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        print(f'   [{split}] {len(self.samples)} samples | augment={self.augment}')

    def _augment(self, image, mask_np):
        """Apply random flip + crop to image AND mask using numpy array."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask_np = np.fliplr(mask_np).copy()

        # Random crop
        H, W = self.img_size
        scale = random.uniform(0.75, 1.0)
        ch, cw = int(H * scale), int(W * scale)
        i = random.randint(0, H - ch)
        j = random.randint(0, W - cw)

        image = TF.crop(image, i, j, ch, cw)
        mask_np = mask_np[i:i + ch, j:j + cw]

        image = image.resize((W, H), Image.BILINEAR)
        mask_np = np.array(
            Image.fromarray(mask_np.astype(np.uint8)).resize((W, H), Image.NEAREST)
        ).astype(np.int64)
        return image, mask_np

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        H, W = self.img_size

        # Load and resize image
        image = Image.open(img_path).convert('RGB').resize((W, H), Image.BILINEAR)

        # Decode mask: grayscale index → training ID
        mask_full = decode_mask(lbl_path)
        mask_pil = Image.fromarray(mask_full.astype(np.uint8))
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.int64)

        if self.augment:
            image, mask_np = self._augment(image, mask_np)

        image_t = TF.normalize(TF.to_tensor(image), MEAN, STD)
        label_t = torch.from_numpy(mask_np)

        return image_t, label_t, img_path

# DataLoader builder
def get_dataloaders(splits, config=CONFIG):
    img_size = (config['img_h'], config['img_w'])

    train_ds = CamVidDataset(splits['train'], split='train', img_size=img_size, augment=True)
    val_ds = CamVidDataset(splits['val'], split='val', img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds
