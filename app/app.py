
""" this is also like app_gui.py but only the difference is that the i havent used pop up 
window for uploading an image , and you have to manualy give the image path to this app.py"""

import argparse
import os
import sys

# Allow running this script directly 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import CONFIG, DEVICE, CLASS_NAMES, CLASS_COLORS, NUM_CLASSES
from src.inference import load_model_for_inference, predict_image, class_coverage_report
from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='CamVid Segmentation — Local Demo App')
    parser.add_argument('--image', type=str, required=True, help='Path to an input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a model checkpoint (default: models/checkpoints/best.pth)')
    parser.add_argument('--no-show', action='store_true', help='Do not open a matplotlib window')
    args = parser.parse_args()

    model = load_model_for_inference(checkpoint_path=args.checkpoint, config=CONFIG, device=DEVICE)
    img_size = (CONFIG['img_h'], CONFIG['img_w'])

    pil_img = Image.open(args.image).convert('RGB')
    pred_mask, pred_rgb, ms = predict_image(pil_img, model, device=DEVICE, img_size=img_size)

    out_dir = ensure_dir(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'predictions'))
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(args.image))[0] + '_pred.png')
    Image.fromarray(pred_rgb).save(out_path)
    print(f'\nPrediction saved to: {out_path}  ({ms:.1f} ms)')

    class_coverage_report(pred_mask, CLASS_NAMES)

    if not args.no_show:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'CamVid Segmentation — {os.path.basename(args.image)}', fontsize=12)

        axes[0].imshow(pil_img.resize((img_size[1], img_size[0])))
        axes[0].set_title('Input')
        axes[0].axis('off')

        axes[1].imshow(pred_rgb)
        axes[1].set_title(f'Prediction ({ms:.1f} ms)')
        axes[1].axis('off')

        patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
                  for i in range(NUM_CLASSES)]
        fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=8)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()


if __name__ == '__main__':
    main()
