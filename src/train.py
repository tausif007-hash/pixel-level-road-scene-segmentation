"""
train.py
Training entry point: builds the model/optimizer/scheduler, runs the
train/validation loop 
"""

import os
import time

import torch
import torch.nn as nn

from src.config import CONFIG, DEVICE, NUM_CLASSES, IGNORE_INDEX
from src.dataset import verify_dataset, get_dataloaders
from src.model import build_model
from src.metrics import AverageMeter, SegmentationMetrics
from src.utils import get_loss_fn, save_checkpoint


def train(config=CONFIG, device=DEVICE):
    """
    Run the full training pipeline and return the training history dict.
    """
    # Data 
    print(f'Locating and verifying dataset at: {config["data_root"]}')
    actual_root, splits = verify_dataset(config['data_root'])
    config['data_root'] = actual_root

    train_loader, val_loader, _train_ds, _val_ds = get_dataloaders(splits, config)

    # Model 
    print(f'Building {config["arch"]}...')
    model = build_model(config['arch'], config['num_classes'],
                        config['pretrained']).to(device)

    # Loss 
    criterion = get_loss_fn(use_weights=config['use_weights'], device=device,
                            num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    # Optimizer / scheduler / scaler 
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    epochs = config['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: (1 - ep / epochs) ** 0.9
    )

    scaler = torch.cuda.amp.GradScaler()

    print(f'✅ Optimizer  : AdamW  lr={config["lr"]}')
    print(f'✅ Scheduler  : Polynomial decay over {epochs} epochs')
    print(f'✅ Scaler     : GradScaler (fresh)')
    print(f'✅ Loss       : {criterion}')
    print()

    #  Training loop 
    best_miou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'pa': [], 'miou': []}

    print(f'Starting training — {epochs} epochs on {device}\n')
    print(f'{"Epoch":>6} {"TrLoss":>8} {"VaLoss":>8} '
          f'{"PixAcc":>8} {"mIoU":>8} {"LR":>10}  {"Time":>6}')
    print('─' * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train 
        model.train()
        train_loss = AverageMeter()

        for images, masks, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)

            if torch.isnan(loss):
                print(f'NaN loss at epoch {epoch} — skipping batch')
                continue

            scaler.scale(loss).backward()

            if config['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
            train_loss.update(loss.item(), images.size(0))

        scheduler.step()

        # Validate 
        model.eval()
        val_loss = AverageMeter()
        metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

        with torch.no_grad():
            for images, masks, _paths in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss.update(loss.item(), images.size(0))
                metrics.update(logits.argmax(dim=1), masks)

        pa, miou, _ = metrics.compute()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss.avg)
        history['val_loss'].append(val_loss.avg)
        history['pa'].append(pa)
        history['miou'].append(miou)

        star = ' ★' if miou > best_miou else ''
        print(f'{epoch:>6} {train_loss.avg:>8.4f} {val_loss.avg:>8.4f} '
              f'{pa*100:>7.2f}% {miou*100:>7.2f}% '
              f'{lr_now:>10.2e}  {elapsed:>5.1f}s{star}')

        # Save last checkpoint → local + "drive" 
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou,
            'arch': config['arch'],
        }
        save_checkpoint(state, config['last_ckpt'],
                        os.path.join(config['drive_ckpt'], 'last.pth'))

        #  Save best checkpoint 
        if miou > best_miou:
            best_miou = miou
            state['best_miou'] = best_miou
            save_checkpoint(state, config['best_ckpt'],
                            os.path.join(config['drive_ckpt'], 'best.pth'))
            print(f'       ★ New best mIoU: {best_miou*100:.2f}% — saved')

    print(f'\nTraining complete!  Best mIoU: {best_miou*100:.2f}%')
    print(f'   Checkpoint saved: {config["best_ckpt"]}')

    return history


if __name__ == '__main__':
    train()
