"""
evaluate.py

this py fileLoads a trained checkpoint and evaluates it on a data split (val or test),
reporting Pixel Accuracy, mIoU, and a per-class IoU table.

( also about the model which i have train on google collab and saved its best path into the drive and uploaded here )
"""

import torch

from src.config import CONFIG, DEVICE, NUM_CLASSES, IGNORE_INDEX
from src.dataset import verify_dataset, CamVidDataset
from torch.utils.data import DataLoader
from src.model import build_model
from src.metrics import AverageMeter, SegmentationMetrics
from src.utils import get_loss_fn, load_checkpoint


def evaluate(checkpoint_path=None, split='test', config=CONFIG, device=DEVICE):
    
    checkpoint_path = checkpoint_path or config['best_ckpt']

    actual_root, splits = verify_dataset(config['data_root'])
    config['data_root'] = actual_root

    img_size = (config['img_h'], config['img_w'])
    eval_ds = CamVidDataset(splits[split], split=split, img_size=img_size, augment=False)
    eval_loader = DataLoader(
        eval_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = build_model(config['arch'], config['num_classes'], pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    criterion = get_loss_fn(use_weights=config['use_weights'], device=device,
                            num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    loss_meter = AverageMeter()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    with torch.no_grad():
        for images, masks, _paths in eval_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)

            loss_meter.update(loss.item(), images.size(0))
            metrics.update(logits.argmax(dim=1), masks)

    pa, miou, per_class_iou = metrics.compute()

    print(f'\n Evaluation on "{split}" split')
    print(f'   Loss        : {loss_meter.avg:.4f}')
    print(f'   Pixel Acc.  : {pa*100:.2f}%')
    print(f'   mIoU        : {miou*100:.2f}%')
    metrics.print_table()

    return pa, miou, per_class_iou


if __name__ == '__main__':
    evaluate()
