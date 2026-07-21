# Semantic Segmentation for Autonomous Driving — CamVid

A production-structured PyTorch project for pixel-level road-scene
segmentation, built around a **U-Net decoder with a pretrained ResNet-34
encoder**, trained on the **CamVid** dataset (11 semantic classes: Sky,
Building, Road, Pedestrian, Car, Bicyclist, etc.).

This project began as a Google Colab notebook. The model was trained and a
best checkpoint (`best.pth`) was already produced there; this repository is
a clean, modular refactor of that notebook into a standard local Python
project — the training/inference behavior is unchanged.

## Features

- U-Net + ResNet-34 encoder (ImageNet-pretrained), with `unet` and
  `unet_small` variants also available for quick CPU testing
- Automatic CamVid layout/label-naming detection and train/val/test
  splitting from `valid.txt`
- Mixed-precision training (`torch.cuda.amp`) with gradient clipping and a
  polynomial learning-rate decay schedule
- Class-imbalance handling via inverse-frequency loss weighting
- Pixel Accuracy + mean IoU (with per-class IoU breakdown) evaluation
- Single-image and batch-folder inference with colorized mask output
- Training-curve and prediction visualizations saved to `outputs/`

## Folder Structure

```
Semantic_Segmentation_CamVid/
│
├── data/                    # CamVid dataset lives here (not committed)
│   └── README.md
│
├── src/                     # Core library code
│   ├── config.py            # paths, hyperparameters, class metadata
│   ├── dataset.py           # download, splits, Dataset, DataLoader
│   ├── model.py              # UNet / UNetSmall / UNetResNet34
│   ├── train.py              # training loop
│   ├── evaluate.py           # evaluation on val/test
│   ├── inference.py          # single-image / folder inference
│   ├── visualization.py      # plots, mask colorization
│   ├── metrics.py            # AverageMeter, SegmentationMetrics
│   └── utils.py              # checkpoint I/O, loss factory, seeding
│
├── app/
│   └── app.py                # local demo: run inference on one image
│
├── notebooks/
│   └── original_notebook.ipynb  # original Colab notebook, for reference
│
├── models/checkpoints/       # trained checkpoints (best.pth, last.pth)
├── outputs/                  # predictions, overlays, plots, logs
│
├── requirements.txt
├── run.py                    # CLI entry point
└── README.md
```

## Installation

Requires Python 3.10+ and (optionally) a CUDA-capable GPU.

```bash
git clone <this-repo>
cd Semantic_Segmentation_CamVid
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Dataset Setup

```bash
python run.py download   # downloads + extracts CamVid into data/camvid/
python run.py verify     # confirms layout and builds train/val/test splits
```

See [`data/README.md`](data/README.md) for the expected folder layout and
manual-download instructions.

## Training

```bash
python run.py train
```

Trains for `CONFIG['epochs']` epochs (default 100) and saves:

- `models/checkpoints/last.pth` — most recent epoch
- `models/checkpoints/best.pth` — best validation mIoU
- `outputs/plots/training_curves.png` — loss / pixel-accuracy / mIoU curves

All hyperparameters live in `src/config.py`.

## Evaluation

```bash
python run.py evaluate --split test
python run.py evaluate --split val --checkpoint models/checkpoints/last.pth
```

Reports loss, Pixel Accuracy, mIoU, and a per-class IoU table.

## Inference

```bash
# Single image
python run.py infer --image path/to/scene.jpg

# Whole folder
python run.py infer --folder path/to/images/

# Or use the demo app (shows a matplotlib window)
python app/app.py --image path/to/scene.jpg
```

Colorized prediction masks are saved to `outputs/predictions/`.

## Results

The best checkpoint achieved strong segmentation performance on the CamVid
validation set (Pixel Accuracy and mIoU as reported during the original
Colab training run — see `outputs/plots/training_curves.png` after training
to reproduce these numbers locally).

## Future Improvements

- Add TTA (test-time augmentation) for inference
- Export to ONNX / TorchScript for deployment
- Add a lightweight web UI (e.g. Gradio) on top of `app/app.py`
- Experiment with additional encoders (EfficientNet, ResNet-50)

## License

Released under the [MIT License](LICENSE).
