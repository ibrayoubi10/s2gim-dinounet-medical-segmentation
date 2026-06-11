"""
GlaS segmentation training pipeline using the DiNOUNet backbone
(DINOv3 ViT-B/16 + UNet decoder).

- Single integrated main: train + per-epoch validation + final test on
  the held-out Test_Folder.
- No data augmentation (CutMix / superpixel / saliency removed).
- No k-fold cross-validation: a deterministic 80/20 split is performed
  on the train folder.
- All outputs (logs, checkpoints, results, metrics CSV) are written under
  ./runs/glas_dino_v1/.
"""

import sys
import os
sys.path.append(os.path.abspath('/data3/nkozah/my_project'))
sys.path.append(os.path.abspath('/data3/nkozah/my_project/glas_dinounet'))

import argparse
import csv
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Glas_dataset import RandomGenerator, ValGenerator, ImageToImage2D_kfold
from stream_metrics_hd95_fast2 import dice_binary_class, IoU_binary_class, hd95Dan
from DiceLoss import BinaryDiceLoss
from dinounet_seg import build_dinounet


# -----------------------------------------------------------------------------
# Run directory and logging
# -----------------------------------------------------------------------------
RUN_DIR = './runs/glas_dino_v1'
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, 'checkpoints'), exist_ok=True)


def make_print_to_file(path=RUN_DIR):
    class Logger(object):
        def __init__(self, filename, path):
            self.terminal = sys.stdout
            os.makedirs(path, exist_ok=True)
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('DAY%Y_%m_%d_') + 'Glas_DiNOUNet.log'
    sys.stdout = Logger(fileName, path=path)


make_print_to_file()


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Glas',
                        choices=['Glas', 'MoNuSeg', 'ISIC2017T1', 'ISIC2017T1_Small'])
    parser.add_argument("--train_dataset", type=str,
                        default='/data3/nkozah/my_project/Data/GlaS/Train_Folder')
    parser.add_argument("--test_dataset", type=str,
                        default='/data3/nkozah/my_project/Data/GlaS/Test_Folder')

    # Pretrained DINOv3 backbone weights
    parser.add_argument("--pretrained_weights", type=str,
                        default='/data3/nkozah/my_project/training_monuseg/'
                                'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

    # Training hyper-parameters (mirrors the DeepLab main)
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0)
    parser.add_argument("--NB_EPOCH", type=int, default=420)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of train_dataset used for validation.")
    parser.add_argument("--loss_type", type=str, default='BCE',
                        choices=['cross_entropy', 'BCE'])
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--gpu_ids", type=list, default=[0])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--freeze_backbone", action='store_true', default=False)
    return parser


# -----------------------------------------------------------------------------
# LR schedule (identical to the DeepLab main)
# -----------------------------------------------------------------------------
def get_lr(epoch, base_lr):
    if epoch <= 4:
        if base_lr >= 1:
            return base_lr * ((epoch + 1) / 5)
        return base_lr
    if epoch <= 149:
        return base_lr
    if epoch <= 199:
        return base_lr / 2
    if epoch <= 249:
        return base_lr / 4
    if epoch <= 279:
        return base_lr / 8
    if epoch <= 309:
        return base_lr / 10
    if epoch <= 329:
        return base_lr / 20
    if epoch <= 349:
        return base_lr / 50
    if epoch <= 369:
        return base_lr / 80
    if epoch <= 399:
        return base_lr / 100
    return base_lr / 1000


# -----------------------------------------------------------------------------
# Evaluation utility (validation / test / train-eval pass)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion_seg, criterion_dice):
    model.eval()
    iou_sum = 0.0
    dice_sum = 0.0
    hd95_sum = 0.0
    hd95_count = 0
    loss_sum = 0.0
    n_batches = 0
    n_samples = 0

    for samples, _ in loader:
        images = samples['image'].to(device, dtype=torch.float32)
        labels = samples['label'].to(device, dtype=torch.float32)

        outputs = model(images)
        loss_seg = criterion_seg(outputs, labels)
        loss_dice = criterion_dice(outputs, labels)
        loss = loss_seg + loss_dice

        loss_sum += loss.item()
        n_batches += 1
        n_samples += images.shape[0]

        dice_sum += dice_binary_class(outputs, labels)
        iou_sum += IoU_binary_class(outputs, labels)

        binary_outputs = (outputs > 0.5).float()
        for j in range(binary_outputs.size(0)):
            if binary_outputs[j].sum() > 0 and labels[j].sum() > 0:
                hd95_sum += hd95Dan(binary_outputs[j].unsqueeze(0),
                                    labels[j].unsqueeze(0))
                hd95_count += 1

    avg_loss = loss_sum / max(n_batches, 1)
    avg_dice = dice_sum / max(n_samples, 1)
    avg_iou = iou_sum / max(n_samples, 1)
    avg_hd95 = hd95_sum / hd95_count if hd95_count > 0 else float('inf')
    return avg_loss, avg_dice, avg_iou, avg_hd95


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    opts = get_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # ----- Train / Val split (no k-fold) -----
    filelists = np.array(sorted(os.listdir(opts.train_dataset + "/img")))
    rng = np.random.RandomState(opts.random_seed)
    perm = rng.permutation(len(filelists))
    n_val = max(1, int(round(opts.val_split * len(filelists))))
    val_filelists = filelists[perm[:n_val]]
    train_filelists = filelists[perm[n_val:]]
    print(f"Total: {len(filelists)} | train: {len(train_filelists)} | val: {len(val_filelists)}")

    train_tf = RandomGenerator(output_size=[opts.img_size, opts.img_size])
    val_tf = ValGenerator(output_size=[opts.img_size, opts.img_size])

    train_dataset = ImageToImage2D_kfold(opts.train_dataset, train_tf,
                                         image_size=opts.img_size,
                                         filelists=train_filelists,
                                         task_name=opts.dataset)
    val_dataset = ImageToImage2D_kfold(opts.train_dataset, val_tf,
                                       image_size=opts.img_size,
                                       filelists=val_filelists,
                                       task_name=opts.dataset)
    # Test set: every file in Test_Folder/img
    test_filelists = np.array(sorted(os.listdir(opts.test_dataset + "/img")))
    test_dataset = ImageToImage2D_kfold(opts.test_dataset, val_tf,
                                        image_size=opts.img_size,
                                        filelists=test_filelists,
                                        task_name=opts.dataset)

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=opts.val_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opts.val_batch_size, shuffle=False)

    # ----- Model -----
    model_name = os.path.join(RUN_DIR, 'checkpoints', 'Glas_DiNOUNet_best.pth')
    pretrained = opts.pretrained_weights if not opts.RESUME else None
    model = build_dinounet(
        num_classes=1,
        img_size=opts.img_size,
        pretrained_weights=pretrained,
        freeze_backbone=opts.freeze_backbone,
    )
    if opts.RESUME and os.path.isfile(model_name):
        print(f"Resuming from {model_name}")
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

    if len(opts.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # ----- Loss -----
    if opts.loss_type == 'cross_entropy':
        criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        criterion_seg = nn.BCELoss(reduction='mean')
    criterion_dice = BinaryDiceLoss()

    # ----- Output files -----
    tm = datetime.datetime.now().strftime('T%m%d%H%M')
    txt_results = os.path.join(RUN_DIR, f'{tm}_Glas_DiNOUNet_results.txt')
    csv_path = os.path.join(RUN_DIR, 'metrics.csv')
    csv_header = ['epoch', 'lr', 'train_loss', 'train_eval_loss',
                  'train_dice', 'train_jaccard', 'train_hd95',
                  'val_dice', 'val_jaccard', 'val_hd95',
                  'best_val_dice_so_far', 'best_epoch_so_far']
    if not opts.RESUME or not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(csv_header)

    best_dice = 0.0
    best_iou = 0.0
    best_epoch = -1

    # ----- Training -----
    for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
        lr = get_lr(epoch, opts.LR)
        optimizer = torch.optim.AdamW(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=opts.weight_decay,
        )

        model.train()
        list_loss, list_loss_seg, list_loss_dice = [], [], []
        for samples, _ in tqdm(train_loader, desc=f"epoch {epoch} train"):
            images = samples['image'].to(device, dtype=torch.float32)
            labels = samples['label'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss_seg = criterion_seg(outputs, labels)
            loss_dice = criterion_dice(outputs, labels)
            loss = loss_seg + loss_dice
            loss.backward()
            optimizer.step()

            list_loss.append(loss.detach())
            list_loss_seg.append(loss_seg.detach())
            list_loss_dice.append(loss_dice.detach())

        train_loss = torch.stack(list_loss).mean().item() if list_loss else 0.0
        train_loss_seg = torch.stack(list_loss_seg).mean().item() if list_loss_seg else 0.0
        train_loss_dice = torch.stack(list_loss_dice).mean().item() if list_loss_dice else 0.0

        # ----- Eval pass on train (clean metrics under model.eval()) -----
        train_eval_loss, train_dice, train_iou, train_hd95 = evaluate(
            model, train_eval_loader, device, criterion_seg, criterion_dice
        )
        # ----- Eval pass on val -----
        val_loss, val_dice, val_iou, val_hd95 = evaluate(
            model, val_loader, device, criterion_seg, criterion_dice
        )

        # ----- Best-model bookkeeping (based on validation Dice) -----
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = epoch
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, model_name)
            print(f"[Best] Epoch {epoch} val_dice={val_dice:.4f} val_iou={val_iou:.4f} (saved)")

        # ----- Console + per-epoch txt log -----
        line = (f"Epoch {epoch} | lr={lr:.2e} | "
                f"train_loss={train_loss:.4f} (seg={train_loss_seg:.4f} dice={train_loss_dice:.4f}) | "
                f"train_eval_loss={train_eval_loss:.4f} | "
                f"train_dice={train_dice:.4f} train_iou={train_iou:.4f} train_hd95={train_hd95:.4f} | "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} val_hd95={val_hd95:.4f} | "
                f"best_val_dice={best_dice:.4f} @ epoch {best_epoch}")
        print(line)
        with open(txt_results, 'a') as f:
            f.write(line + "\n")

        # ----- CSV row -----
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}",
                f"{train_loss:.6f}", f"{train_eval_loss:.6f}",
                f"{train_dice:.6f}", f"{train_iou:.6f}", f"{train_hd95:.6f}",
                f"{val_dice:.6f}", f"{val_iou:.6f}", f"{val_hd95:.6f}",
                f"{best_dice:.6f}", best_epoch,
            ])

    torch.cuda.empty_cache()

    # ----- Final test on Test_Folder using the best checkpoint -----
    print("\n=========== Final test on Test_Folder ===========")
    if os.path.isfile(model_name):
        state = torch.load(model_name, map_location='cpu')
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        print(f"Loaded best checkpoint: {model_name}")

    test_loss, test_dice, test_iou, test_hd95 = evaluate(
        model, test_loader, device, criterion_seg, criterion_dice
    )
    test_line = (f"TEST | best_epoch={best_epoch} | best_val_dice={best_dice:.4f} | "
                 f"test_loss={test_loss:.4f} | test_dice={test_dice:.4f} | "
                 f"test_iou={test_iou:.4f} | test_hd95={test_hd95:.4f}")
    print(test_line)
    with open(txt_results, 'a') as f:
        f.write("\n" + test_line + "\n")
    with open(os.path.join(RUN_DIR, 'test_results.txt'), 'w') as f:
        f.write(test_line + "\n")


if __name__ == '__main__':
    main()
