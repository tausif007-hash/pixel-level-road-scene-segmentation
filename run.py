"""
run.py
======
Single entry point for the CamVid semantic segmentation project.

Usage (from the project root, in a terminal / VS Code):

    python run.py download                 # download + extract CamVid
    python run.py train                     # train the model (Section 5)
    python run.py evaluate --split test     # evaluate best checkpoint
    python run.py infer --image path.jpg    # run inference on one image
    python run.py infer --folder path/dir   # run inference on a folder

All behaviour (hyperparameters, architecture, preprocessing, checkpoint
logic) is unchanged from the original Colab notebook — this script only
wires together the modules in `src/`.
"""

import argparse
import os

from src.config import CONFIG, DEVICE
from src.dataset import download_camvid, verify_dataset
from src.train import train
from src.evaluate import evaluate
from src.inference import load_model_for_inference, predict_image, predict_folder, class_coverage_report
from src.visualization import plot_training_curves
from src.utils import ensure_dir
from PIL import Image


def cmd_download(args):
    download_camvid(CONFIG['data_root'])


def cmd_verify(args):
    verify_dataset(CONFIG['data_root'])


def cmd_train(args):
    history = train(config=CONFIG, device=DEVICE)
    plots_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'plots')
    ensure_dir(plots_dir)
    plot_training_curves(history, save_path=os.path.join(plots_dir, 'training_curves.png'))


def cmd_evaluate(args):
    evaluate(checkpoint_path=args.checkpoint, split=args.split, config=CONFIG, device=DEVICE)


def cmd_infer(args):
    model = load_model_for_inference(checkpoint_path=args.checkpoint, config=CONFIG, device=DEVICE)
    img_size = (CONFIG['img_h'], CONFIG['img_w'])

    if args.image:
        pil_img = Image.open(args.image).convert('RGB')
        pred_mask, pred_rgb, ms = predict_image(pil_img, model, device=DEVICE, img_size=img_size)

        out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'predictions')
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(args.image))[0] + '_pred.png')
        Image.fromarray(pred_rgb).save(out_path)

        print(f'✅ Prediction saved to: {out_path}  ({ms:.1f} ms)')
        from src.config import CLASS_NAMES
        class_coverage_report(pred_mask, CLASS_NAMES)

    elif args.folder:
        out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'predictions')
        predict_folder(args.folder, model, device=DEVICE, img_size=img_size, output_dir=out_dir)

    else:
        raise ValueError('Provide either --image <path> or --folder <path> for inference.')


def build_parser():
    parser = argparse.ArgumentParser(description='CamVid Semantic Segmentation — U-Net + ResNet-34')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('download', help='Download and extract the CamVid dataset').set_defaults(func=cmd_download)
    sub.add_parser('verify', help='Verify dataset layout and build splits').set_defaults(func=cmd_verify)
    sub.add_parser('train', help='Train the model').set_defaults(func=cmd_train)

    p_eval = sub.add_parser('evaluate', help='Evaluate a checkpoint')
    p_eval.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: best.pth)')
    p_eval.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser('infer', help='Run inference on an image or folder')
    p_infer.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: best.pth)')
    p_infer.add_argument('--image', type=str, default=None, help='Path to a single image')
    p_infer.add_argument('--folder', type=str, default=None, help='Path to a folder of images')
    p_infer.set_defaults(func=cmd_infer)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
