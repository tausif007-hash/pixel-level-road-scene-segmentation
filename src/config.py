"""config.py
All paths, hyperparameters, and constants are given in this file 
"""

import os
import torch


# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local project folders
DATA_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'data', 'camvid')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'models', 'checkpoints')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')
OVERLAYS_DIR = os.path.join(OUTPUTS_DIR, 'overlays')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(OVERLAYS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# Global config dict 
CONFIG = {
    # Data
    'data_root': DATA_ROOT_DEFAULT,
    'img_h': 360,          # resize height  (original (720))
    'img_w': 480,          # resize width   (original (960))

    # Model
    'arch': 'unet_resnet34', 
    'num_classes': 11,
    'pretrained': True,

    # Training
    'epochs': 100,
    'batch_size': 8,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'use_weights': True,    #imp
    'grad_clip': 1.0,

    # Checkpoints
    'local_ckpt': CKPT_DIR,
    'drive_ckpt': CKPT_DIR,   
    'best_ckpt': os.path.join(CKPT_DIR, 'best.pth'),
    'last_ckpt': os.path.join(CKPT_DIR, 'last.pth'),
}


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CamVid dataset download URL 
CAMVID_URL = 'https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz'

# Raw index (from CamVid codes.txt, 0-31) → training class ID (0-10).
# Everything else is mapped to ignore_index

RAW_TO_TRAIN = {
    21: 0,   # Sky
    4: 1,    # Building
    8: 2,    # Column_Pole
    17: 3,   # Road
    19: 4,   # Sidewalk
    26: 5,   # Tree
    20: 6,   # SignSymbol
    9: 7,    # Fence
    5: 8,    # Car
    16: 9,   # Pedestrian
    2: 10,   # bicyclist
}

NUM_CLASSES = 11
IGNORE_INDEX = 11   # all other raw classes → ignored

CLASS_NAMES = [
    'Sky', 'Building', 'Column_Pole', 'Road', 'Sidewalk',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
]

# Colours for visualization 
import numpy as np  # noqa: E402  

CLASS_COLORS = np.array([
    [128, 128, 128],   # Sky
    [128, 0, 0],       # Building
    [192, 192, 128],   # Column_Pole
    [128, 64, 128],    # Road
    [60, 40, 222],     # Sidewalk
    [128, 128, 0],     # Tree
    [192, 128, 128],   # SignSymbol
    [64, 64, 128],     # Fence
    [64, 0, 128],      # Car
    [64, 64, 0],       # Pedestrian
    [0, 128, 192],     # Bicyclist
], dtype=np.uint8)


CONFIG['num_classes'] = NUM_CLASSES

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Random seed used for reproducible train/val/test splits
SEED = 42

# Approximate inverse-frequency class weights 
CLASS_PIXEL_FREQ = np.array([
    17.6,   # Sky
    29.1,   # Building
    1.0,    # Column_Pole
    23.4,   # Road
    4.5,    # Sidewalk
    13.2,   # Tree
    0.7,    # SignSymbol
    2.0,    # Fence
    6.5,    # Car
    0.9,    # Pedestrian
    1.1,    # Bicyclist
])
