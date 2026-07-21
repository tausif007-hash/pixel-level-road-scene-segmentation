"""
inference.py
Loading a trained checkpoint and running inference on a single image 
"""

import os
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.config import CONFIG, DEVICE, MEAN, STD, PREDICTIONS_DIR
from src.model import build_model
from src.utils import load_checkpoint
from src.visualization import colorize_mask


def load_model_for_inference(checkpoint_path=None, config=CONFIG, device=DEVICE):
    #Build the model architecture and load trained weights for inference.
    checkpoint_path = checkpoint_path or config['best_ckpt']
    model = build_model(config['arch'], config['num_classes'], pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f'   Model ready for inference on {device}')
    return model


@torch.no_grad()
def predict_image(pil_img, model, device=DEVICE, img_size=(360, 480)):
    """
    Segment a single PIL image.

    Returns:
        pred_mask : [H, W] integer tensor
        pred_rgb  : [H, W, 3] coloured numpy array
        ms        : inference time in milliseconds
    """
    H, W = img_size
    resized = pil_img.resize((W, H), Image.BILINEAR)
    t = TF.to_tensor(resized)
    t = TF.normalize(t, MEAN, STD)
    t = t.unsqueeze(0).to(device)

    t0 = time.time()
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        logits = model(t)
    ms = (time.time() - t0) * 1000

    pred_mask = logits.argmax(dim=1)[0].cpu()
    pred_rgb = colorize_mask(pred_mask.numpy())
    return pred_mask, pred_rgb, ms


def predict_folder(input_dir, model, device=DEVICE, img_size=(360, 480),
                    output_dir=PREDICTIONS_DIR):
    
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    results = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert('RGB')

        _pred_mask, pred_rgb, ms = predict_image(pil_img, model, device=device, img_size=img_size)

        out_name = os.path.splitext(fname)[0] + '_pred.png'
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(pred_rgb).save(out_path)

        print(f'   {fname:<30} → {out_name}  ({ms:.1f} ms)')
        results.append((img_path, out_path, ms))

    return results


def class_coverage_report(pred_mask, class_names):
    """
    Printing  a text bar-chart of the percentage of the image covered by each
    predicted class .
    """
    arr = pred_mask.numpy()
    total = arr.size
    print('\nClass coverage:')
    for i, name in enumerate(class_names):
        pct = (arr == i).sum() / total * 100
        if pct > 0.1:
            bar = '█' * int(pct / 2)
            print(f'  {name:<14} {pct:>5.1f}%  {bar}')
