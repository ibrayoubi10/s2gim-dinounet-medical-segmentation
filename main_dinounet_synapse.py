"""
Synapse multi-organ segmentation pipeline using the DiNOUNet backbone
(DINOv3 ViT-B/16 + UNet decoder), built on the same skeleton as
`main_TransUnet_synapse.py` but swapping the network for DiNOUNet.

Differences vs. the binary GlaS / MoNuSeg DiNOUNet mains:
- Multi-class output (num_classes=9): the network is built with
  `apply_sigmoid=False` so it returns raw logits (B, C, H, W) suitable
  for `CrossEntropyLoss` and a softmax dice loss.
- The Synapse images are single-channel grayscale CT slices: we repeat
  the channel 1->3 inside `_prep_2d_batch` so the DINOv3 ViT-B/16 (which
  expects 3 channels) accepts them.
- Test split is `test_vol` evaluated as 2D slices via `TestVol2DSliceDataset`.
- Per-class + macro Dice / Jaccard / HD95 are tracked through
  `Multilabel_metrics`.
- CSV history mirrors the TransUNet/Synapse main (macro + per-class
  metrics for both train and test).
- Early stopping with `--es_patience` / `--es_min_delta`.

All outputs are written under `./runs/synapse_dino_v1/`.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---- Project paths ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
sys.path.append(os.path.abspath('/data3/nkozah/my_project'))
sys.path.append(os.path.abspath('/data3/nkozah/my_project/Ibrahim'))

from Datasets.synapse_dataset_config import SynapseDataset, RandomGenerator
from metrics.Multilabel_metrics import init_running_metrics, update_running_metrics, finalize_metrics
from dinounet_seg import build_dinounet


# ============================================================
# Loss (multi-class soft Dice with optional background exclusion)
# ============================================================
class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss over softmax(logits) for multi-class segmentation.

    Mirrors the interface of the TransUNet/Synapse main:
        dice_loss = MultiClassDiceLoss(include_background=False)
    """

    def __init__(self, include_background: bool = False, smooth: float = 1.0):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W)
        # target : (B, H, W) int64
        n_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=n_classes)
        # (B, H, W, C) -> (B, C, H, W)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_onehot = target_onehot[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


# ============================================================
# Utils (taken from main_TransUnet_synapse.py)
# ============================================================
def estimate_ce_weights_from_loader(train_loader, num_classes, device, max_batches=200,
                                    clamp_min=0.1, clamp_max=10.0):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for k, s in enumerate(train_loader):
        if k >= max_batches:
            break
        y = s["label"].view(-1)
        binc = torch.bincount(y, minlength=num_classes).double()
        counts += binc
    freq = counts / counts.sum().clamp_min(1.0)
    w = 1.0 / (freq + 1e-12)
    w = w / w.mean().clamp_min(1e-12)
    w = w.float()
    w = torch.clamp(w, clamp_min, clamp_max).to(device)
    return w, counts


def append_csv(csv_path, header, row):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(row)


def _prep_2d_batch(images, labels, device):
    """
    Accepts:
      - images (B, H, W)    -> (B, 1, H, W) -> (B, 3, H, W)  [repeat for ViT]
      - images (B, 1, H, W) -> (B, 3, H, W)
      - images (B, 3, H, W) -> kept as is
    DINOv3 ViT-B/16 requires 3 input channels.
    """
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    imgs = images.to(device, dtype=torch.float32, non_blocking=True)
    gts = labels.to(device, dtype=torch.long, non_blocking=True)
    return imgs, gts


def _ensure_depth_first(vol_np):
    if vol_np.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol_np.shape}")
    depth_axis = int(np.argmin(list(vol_np.shape)))
    if depth_axis != 0:
        vol_np = np.moveaxis(vol_np, depth_axis, 0)
    return vol_np


def _nan_to_empty(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _per_class_to_arrays(per_class, C):
    dice = [np.nan] * C
    jac = [np.nan] * C
    hd95 = [np.nan] * C
    if isinstance(per_class, dict):
        for k, v in per_class.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if cid < 0 or cid >= C:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                dice[cid], jac[cid], hd95[cid] = v[0], v[1], v[2]
            elif isinstance(v, dict):
                dice[cid] = v.get("dice", np.nan)
                jac[cid] = v.get("jac", np.nan)
                hd95[cid] = v.get("hd95", np.nan)
    elif isinstance(per_class, (list, tuple)):
        for cid in range(min(C, len(per_class))):
            v = per_class[cid]
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                dice[cid], jac[cid], hd95[cid] = v[0], v[1], v[2]
            elif isinstance(v, dict):
                dice[cid] = v.get("dice", np.nan)
                jac[cid] = v.get("jac", np.nan)
                hd95[cid] = v.get("hd95", np.nan)
    return dice, jac, hd95


# ============================================================
# Early stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        assert mode in ("max", "min")
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "max":
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def step(self, current):
        if current is None or (isinstance(current, float) and np.isnan(current)):
            self.bad_epochs += 1
            return False, (self.bad_epochs >= self.patience)
        if self._is_improvement(float(current)):
            self.best = float(current)
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, (self.bad_epochs >= self.patience)


# ============================================================
# Test-vol -> slice dataset (same logic as TransUNet/Synapse main)
# ============================================================
class TestSliceTransform:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        if (x, y) != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(label.astype(np.int64))                 # (H, W)
        return {"image": image, "label": label}


class TestVol2DSliceDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        self.index = []
        for case_idx in range(len(self.base)):
            s = self.base[case_idx]
            img = _ensure_depth_first(np.asarray(s["image"]))
            lab = _ensure_depth_first(np.asarray(s["label"]))
            if img.shape[0] != lab.shape[0]:
                raise ValueError(f"Depth mismatch: {img.shape} vs {lab.shape}")
            for z in range(img.shape[0]):
                self.index.append((case_idx, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        case_idx, z = self.index[i]
        s = self.base[case_idx]
        img = _ensure_depth_first(np.asarray(s["image"]))
        lab = _ensure_depth_first(np.asarray(s["label"]))
        sample = {
            "image": img[z],
            "label": lab[z],
            "case_name": s.get("case_name", str(case_idx)),
            "slice_idx": z,
        }
        if self.transform is not None:
            t = self.transform({"image": sample["image"], "label": sample["label"]})
            sample["image"], sample["label"] = t["image"], t["label"]
        return sample


def evaluate_on_test_2d(model, test_loader, num_classes, device, ce_loss, dice_loss):
    model.eval()
    running = init_running_metrics(num_classes)

    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for samples in tqdm(test_loader, total=len(test_loader), desc="TEST_2D(test_vol->slices)"):
            images = samples["image"]
            labels = samples["label"]
            imgs, gts = _prep_2d_batch(images, labels, device)

            logits = model(imgs)  # (B, C, H, W)
            loss_ce = ce_loss(logits, gts)
            loss_d = dice_loss(logits, gts)
            loss = loss_ce + loss_d

            loss_sum += float(loss.detach().cpu())
            ce_sum += float(loss_ce.detach().cpu())
            dice_sum += float(loss_d.detach().cpu())
            n_batches += 1

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            pred_np = pred.detach().cpu().numpy()
            gt_np = gts.detach().cpu().numpy()

            for b in range(pred_np.shape[0]):
                running = update_running_metrics(running, pred_np[b], gt_np[b], num_classes)

    per_class, macro = finalize_metrics(running, num_classes)
    test_loss = loss_sum / max(1, n_batches)
    test_ce = ce_sum / max(1, n_batches)
    test_dice_l = dice_sum / max(1, n_batches)
    return per_class, macro, (test_loss, test_ce, test_dice_l)


# ============================================================
# Args / main
# ============================================================
def get_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--root_dir", type=str, default="/data3/nkozah/my_project/Data/synapse")
    p.add_argument("--num_classes", type=int, default=9)

    # Pretrained DINOv3 backbone weights
    p.add_argument("--pretrained_weights", type=str,
                   default="/data3/nkozah/my_project/glas_monuseg_dinounet/"
                           "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    p.add_argument("--freeze_backbone", action="store_true", default=False)

    # Training hyper-parameters
    p.add_argument("--NB_EPOCH", type=int, default=200)
    p.add_argument("--LR", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=8)

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--exp_name", type=str, default="synapse_dino_v1")

    # compute train metrics every N iters (keeps training fast)
    p.add_argument("--metrics_every", type=int, default=20)

    p.add_argument("--save_best_on", type=str, default="macro_dice", choices=["macro_dice", "loss"])
    p.add_argument("--random_seed", type=int, default=1234)

    # CE weights
    p.add_argument("--ce_weight_batches", type=int, default=200)
    p.add_argument("--ce_w_min", type=float, default=0.1)
    p.add_argument("--ce_w_max", type=float, default=10.0)

    # EarlyStopping
    p.add_argument("--early_stop", type=int, default=1, choices=[0, 1])
    p.add_argument("--es_patience", type=int, default=25)
    p.add_argument("--es_min_delta", type=float, default=1e-4)

    p.add_argument("--out_root", type=str, default="./experiments")
    return p


def main():
    opts = get_argparser().parse_args()

    random.seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    exp_dir = os.path.join(opts.out_root, opts.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, "history.csv")

    C = opts.num_classes

    # -------------------------
    # CSV header (macro + per-class)
    # -------------------------
    header_cols = [
        "epoch", "lr",
        "train_loss", "train_ce", "train_dice_loss",
        "train_macro_dice", "train_macro_jac", "train_macro_hd95",
    ]
    for c in range(C):
        header_cols += [f"train_dice_c{c}", f"train_jac_c{c}", f"train_hd95_c{c}"]

    header_cols += [
        "test_loss", "test_ce", "test_dice_loss",
        "test_macro_dice", "test_macro_jac", "test_macro_hd95",
    ]
    for c in range(C):
        header_cols += [f"test_dice_c{c}", f"test_jac_c{c}", f"test_hd95_c{c}"]

    header = ",".join(header_cols) + "\n"

    # -------------------------
    # Datasets / loaders
    # -------------------------
    tr_transform = RandomGenerator((opts.img_size, opts.img_size))
    train_ds = SynapseDataset(opts.root_dir, split="train", transform=tr_transform,
                              strict=True, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    test_base = SynapseDataset(opts.root_dir, split="test_vol", transform=None,
                               strict=True, verbose=False)
    test_ds = TestVol2DSliceDataset(test_base,
                                    transform=TestSliceTransform((opts.img_size, opts.img_size)))
    test_loader = DataLoader(test_ds, batch_size=opts.test_batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print("Train samples:", len(train_ds))
    print("Test volumes :", len(test_base))
    print("Test slices  :", len(test_ds))

    # -------------------------
    # Model (DiNOUNet, multi-class -> raw logits)
    # -------------------------
    model = build_dinounet(
        num_classes=C,
        img_size=opts.img_size,
        pretrained_weights=opts.pretrained_weights,
        freeze_backbone=opts.freeze_backbone,
        apply_sigmoid=False,
    ).to(device)

    # -------------------------
    # Losses + optimizer
    # -------------------------
    ce_w, counts = estimate_ce_weights_from_loader(
        train_loader, C, device,
        max_batches=opts.ce_weight_batches,
        clamp_min=opts.ce_w_min,
        clamp_max=opts.ce_w_max,
    )
    print("Pixel counts:", counts.cpu().numpy().astype(np.int64))
    print("CE weights  :", ce_w.detach().cpu().numpy())

    ce_loss = nn.CrossEntropyLoss(weight=ce_w, reduction="mean")
    dice_loss = MultiClassDiceLoss(include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.LR, weight_decay=opts.weight_decay)

    last_path = os.path.join(ckpt_dir, "model_last.pth")
    best_path = os.path.join(ckpt_dir, "model_best.pth")

    # best logic + early stopping setup
    if opts.save_best_on == "macro_dice":
        best_score = -1e9
        es = EarlyStopping(patience=opts.es_patience, min_delta=opts.es_min_delta, mode="max")
    else:
        best_score = 1e9
        es = EarlyStopping(patience=opts.es_patience, min_delta=opts.es_min_delta, mode="min")

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(opts.NB_EPOCH):
        lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 72)
        print(f"Epoch {epoch}/{opts.NB_EPOCH - 1} | lr={lr:.6g}")
        print("=" * 72)

        model.train()
        running_train = init_running_metrics(C)

        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        n_batches = 0

        for i, samples in enumerate(tqdm(train_loader, desc="TRAIN_2D", total=len(train_loader))):
            images = samples["image"]
            labels = samples["label"]
            imgs, gts = _prep_2d_batch(images, labels, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)  # (B, C, H, W)

            loss_ce = ce_loss(logits, gts)
            loss_d = dice_loss(logits, gts)
            loss = loss_ce + loss_d

            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            ce_sum += float(loss_ce.detach().cpu())
            dice_sum += float(loss_d.detach().cpu())
            n_batches += 1

            # compute metrics sparsely (keeps train fast)
            if opts.metrics_every > 0 and (i % opts.metrics_every == 0):
                with torch.no_grad():
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    pred_np = pred.detach().cpu().numpy()
                    gt_np = gts.detach().cpu().numpy()
                    for b in range(pred_np.shape[0]):
                        running_train = update_running_metrics(running_train, pred_np[b], gt_np[b], C)

        train_loss = loss_sum / max(1, n_batches)
        train_ce = ce_sum / max(1, n_batches)
        train_dice_l = dice_sum / max(1, n_batches)

        tr_per_class, tr_macro = finalize_metrics(running_train, C)
        tr_macro_dice, tr_macro_jac, tr_macro_hd95 = tr_macro
        tr_dice_arr, tr_jac_arr, tr_hd95_arr = _per_class_to_arrays(tr_per_class, C)

        print("\n--- Train Summary ---")
        print(f"[TRAIN] Loss={train_loss:.6f} | CE={train_ce:.6f} | DiceLoss={train_dice_l:.6f}")
        print(f"[TRAIN] Macro Dice={tr_macro_dice:.4f} | Macro Jac={tr_macro_jac:.4f} | Macro HD95={tr_macro_hd95:.4f}")

        te_per_class, te_macro, te_losses = evaluate_on_test_2d(
            model, test_loader, C, device, ce_loss, dice_loss
        )
        te_macro_dice, te_macro_jac, te_macro_hd95 = te_macro
        test_loss, test_ce, test_dice_l = te_losses
        te_dice_arr, te_jac_arr, te_hd95_arr = _per_class_to_arrays(te_per_class, C)

        print("\n--- Test Summary (test_vol -> 2D slices) ---")
        print(f"[TEST]  Loss={test_loss:.6f} | CE={test_ce:.6f} | DiceLoss={test_dice_l:.6f}")
        print(f"[TEST]  Macro Dice={te_macro_dice:.4f} | Macro Jac={te_macro_jac:.4f} | Macro HD95={te_macro_hd95:.4f}")

        # checkpoints (always save last)
        torch.save(model.state_dict(), last_path)
        print("Saved last checkpoint:", last_path)

        # CSV row
        row = []
        row += [str(epoch), str(lr)]
        row += [str(train_loss), str(train_ce), str(train_dice_l)]
        row += [_nan_to_empty(tr_macro_dice), _nan_to_empty(tr_macro_jac), _nan_to_empty(tr_macro_hd95)]
        for c in range(C):
            row += [_nan_to_empty(tr_dice_arr[c]), _nan_to_empty(tr_jac_arr[c]), _nan_to_empty(tr_hd95_arr[c])]

        row += [str(test_loss), str(test_ce), str(test_dice_l)]
        row += [_nan_to_empty(te_macro_dice), _nan_to_empty(te_macro_jac), _nan_to_empty(te_macro_hd95)]
        for c in range(C):
            row += [_nan_to_empty(te_dice_arr[c]), _nan_to_empty(te_jac_arr[c]), _nan_to_empty(te_hd95_arr[c])]

        append_csv(csv_path, header, ",".join(row) + "\n")

        # best model logic
        if opts.save_best_on == "macro_dice":
            score = te_macro_dice
            if (score is not None) and (not (isinstance(score, float) and np.isnan(score))) and score > best_score:
                best_score = float(score)
                torch.save(model.state_dict(), best_path)
                print(f"Saved best (macro_dice={best_score:.4f}) -> {best_path}")
        else:
            score = train_loss
            if float(score) < best_score:
                best_score = float(score)
                torch.save(model.state_dict(), best_path)
                print(f"Saved best (train_loss={best_score:.6f}) -> {best_path}")

        # early stopping
        if opts.early_stop == 1:
            monitor = te_macro_dice if opts.save_best_on == "macro_dice" else train_loss
            improved, should_stop = es.step(monitor)
            print(f"EarlyStopping: improved={improved} | bad_epochs={es.bad_epochs}/{es.patience} | best={es.best}")
            if should_stop:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
