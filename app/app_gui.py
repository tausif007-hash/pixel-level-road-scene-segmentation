
import argparse
import os
import sys

# allow running of this script 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import CONFIG, DEVICE, CLASS_NAMES, CLASS_COLORS, NUM_CLASSES
from src.inference import load_model_for_inference, predict_image, class_coverage_report
from src.utils import ensure_dir


def show_prediction_window(pil_img, pred_rgb, ms, img_size, image_name):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'CamVid Segmentation — {image_name}', fontsize=12)

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


def run_on_image(image_path, model, img_size, out_dir):
    pil_img = Image.open(image_path).convert('RGB')
    pred_mask, pred_rgb, ms = predict_image(pil_img, model, device=DEVICE, img_size=img_size)

    out_name = os.path.splitext(os.path.basename(image_path))[0] + '_pred.png'
    out_path = os.path.join(out_dir, out_name)
    Image.fromarray(pred_rgb).save(out_path)

    print(f'\n Prediction saved to: {out_path}  ({ms:.1f} ms)')
    class_coverage_report(pred_mask, CLASS_NAMES)

    show_prediction_window(pil_img, pred_rgb, ms, img_size, os.path.basename(image_path))


def main():
    parser = argparse.ArgumentParser(description='CamVid Segmentation — Upload & Predict GUI')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a model checkpoint (default: models/checkpoints/best.pth)')
    args = parser.parse_args()

    print('Loading model — this happens once, may take a few seconds...')
    model = load_model_for_inference(checkpoint_path=args.checkpoint, config=CONFIG, device=DEVICE)
    img_size = (CONFIG['img_h'], CONFIG['img_w'])
    out_dir = ensure_dir(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'predictions'))

    # small window for uploading images 
    root = tk.Tk()
    root.title('CamVid Segmentation')
    root.geometry('340x140')
    root.resizable(False, False)

    status = tk.StringVar(value='Model loaded, Click below to upload an image.')

    def on_upload():
        path = filedialog.askopenfilename(
            title='Select a driving-scene image',
            filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp'), ('All files', '*.*')],
        )
        if not path:
            return
        status.set(f'Running inference on:\n{os.path.basename(path)} ...')
        root.update_idletasks()
        try:
            run_on_image(path, model, img_size, out_dir)
            status.set('Done! Pick another image, or close this window.')
        except Exception as e:
            messagebox.showerror('Error', str(e))
            status.set('Something went wrong — see console for details.')

    tk.Label(root, textvariable=status, wraplength=300, justify='center').pack(pady=15)
    tk.Button(root, text='Upload Image', command=on_upload,
             width=20, height=2).pack(pady=5)

    root.mainloop()


if __name__ == '__main__':
    main()
