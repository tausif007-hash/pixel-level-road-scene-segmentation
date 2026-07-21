
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from src.config import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, MEAN, STD

MEAN_NP = np.array(MEAN)
STD_NP = np.array(STD)


# Core conversions
def colorize_mask(mask_np):
    """2D label array [H,W] → RGB image [H,W,3]"""
    H, W = mask_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        rgb[mask_np == cls_id] = CLASS_COLORS[cls_id]
    return rgb


def denormalize(img_tensor):
    """FloatTensor [3,H,W] → uint8 numpy [H,W,3]"""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD_NP + MEAN_NP
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _legend_patches():
    return [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]


# Class legend 
def plot_class_legend(save_path=None):
    """Visualize the CamVid class-colour legend as a coloured strip."""
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.set_xlim(0, NUM_CLASSES)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle(f'CamVid — {NUM_CLASSES} Training Classes', fontsize=12, y=1.1)

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        rect = plt.Rectangle([i, 0.3], 0.9, 0.5, color=np.array(color) / 255.0)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.1, name, ha='center', va='bottom', fontsize=7, rotation=30)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()


# Single-sample prediction plot 
def plot_prediction(image_tensor, gt_mask, pred_mask, title='', save_path=None):
    """Side-by-side: Input | Ground Truth | Prediction"""
    img_np = denormalize(image_tensor)
    pred_np = colorize_mask(pred_mask.cpu().numpy())
    n = 3 if gt_mask is not None else 2

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(img_np)
    axes[0].set_title('Input')
    axes[0].axis('off')

    if gt_mask is not None:
        axes[1].imshow(colorize_mask(gt_mask.cpu().numpy()))
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        pred_ax = axes[2]
    else:
        pred_ax = axes[1]

    pred_ax.imshow(pred_np)
    pred_ax.set_title('Prediction')
    pred_ax.axis('off')

    fig.legend(handles=_legend_patches(), loc='lower center', ncol=6,
               fontsize=7, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()

# Training curves 
def plot_training_curves(history, save_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    eps = range(1, len(history['train_loss']) + 1)

    axes[0].plot(eps, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(eps, history['val_loss'], label='Val', color='orange')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, [p * 100 for p in history['pa']], color='green')
    axes[1].set_title('Pixel Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps, [m * 100 for m in history['miou']], color='crimson')
    axes[2].set_title('mIoU (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Training Curves — CamVid Segmentation', fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()

# Dataset sample preview 
def visualize_dataset_samples(pairs, n=4, save_path=None):

    import random
    sample_pairs = random.sample(pairs, min(n, len(pairs)))

    fig, axes = plt.subplots(2, len(sample_pairs), figsize=(4.5 * len(sample_pairs), 6))
    fig.suptitle('CamVid — Sample Images (top) and Label Masks (bottom)',
                 fontsize=13, fontweight='bold')

    if len(sample_pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, (img_path, lbl_path) in enumerate(sample_pairs):
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('RGB')

        axes[0, col].imshow(img)
        axes[0, col].set_title(os.path.basename(img_path)[:22], fontsize=7)
        axes[0, col].axis('off')

        axes[1, col].imshow(lbl)
        axes[1, col].set_title('Label mask', fontsize=7)
        axes[1, col].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()


# Multi-sample 
def visualize_predictions_grid(sample_images, sample_labels, predict_fn,
                                img_size, model, save_path=None):
    
    fig, axes = plt.subplots(len(sample_images), 3,
                              figsize=(18, 4.5 * len(sample_images)))
    fig.suptitle('CamVid Predictions — Input | Ground Truth | Prediction', fontsize=13)

    if len(sample_images) == 1:
        axes = [axes]

    for row, img_path in enumerate(sample_images):
        fname = os.path.basename(img_path)
        lbl_path = sample_labels[row]

        pil_img = Image.open(img_path).convert('RGB')

        gt_gray = Image.open(lbl_path).convert('L')
        gt_label_np = np.array(gt_gray)
        gt_rgb_colored = colorize_mask(gt_label_np)
        gt_rgb_colored = Image.fromarray(gt_rgb_colored).resize(
            (img_size[1], img_size[0]), Image.NEAREST)

        pred_mask, pred_rgb, ms = predict_fn(pil_img, model, img_size=img_size)

        axes[row][0].imshow(pil_img.resize((img_size[1], img_size[0])))
        axes[row][0].set_title(f'Input: {fname[:25]}')
        axes[row][0].axis('off')

        axes[row][1].imshow(gt_rgb_colored)
        axes[row][1].set_title('Ground Truth')
        axes[row][1].axis('off')

        axes[row][2].imshow(pred_rgb)
        axes[row][2].set_title(f'Prediction ({ms:.1f} ms)')
        axes[row][2].axis('off')

    fig.legend(handles=_legend_patches(), loc='lower center', ncol=6,
               fontsize=8, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()
