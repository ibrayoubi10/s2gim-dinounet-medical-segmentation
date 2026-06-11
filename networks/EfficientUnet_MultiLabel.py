# Ibrahim/networks/EfficientUNet_EfficientNet_MultiLabel.py

from __future__ import annotations
from collections import OrderedDict
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Ibrahim.networks.EfficientUnet import EfficientNet


# ----------------------------
# Basic UNet blocks
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsample + concat skip + DoubleConv.
    (We use interpolate => no constraint on in channels for upsampling.)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ----------------------------
# Skip collection for YOUR EfficientNet encoder
# ----------------------------
@torch.no_grad()
def get_efficientnet_skips_by_hw(encoder: nn.Module, x: torch.Tensor) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Collect feature maps (B,C,H,W) by spatial resolution (H,W) from encoder forward hooks.

    - We keep LAST occurrence for each (H,W)
    - We DO NOT depend on exact layer names except we filter by module.name prefix 'blocks_' and 'head_swish'
      which matches your EfficientNet.encoder() implementation.
    """
    feats: Dict[Tuple[int, int], torch.Tensor] = OrderedDict()
    hooks = []

    def register_hook(m: nn.Module):
        def hook(_m, _inp, out):
            if not (torch.is_tensor(out) and out.ndim == 4):
                return

            # your encoder uses modules with .name attributes (BatchNorm2d/Swish wrappers)
            name = getattr(_m, "name", "")

            # Capture: block outputs BN + head_swish
            # This keeps compatibility with your existing naming scheme.
            if isinstance(name, str) and (name.startswith("blocks_") or name == "head_swish"):
                hw = (out.shape[-2], out.shape[-1])
                feats[hw] = out

        # avoid container modules
        if not isinstance(m, (nn.Sequential, nn.ModuleList)) and m is not encoder:
            hooks.append(m.register_forward_hook(hook))

    encoder.apply(register_hook)
    _ = encoder(x)
    for h in hooks:
        h.remove()

    if len(feats) < 2:
        raise RuntimeError(
            "Not enough EfficientNet features captured. "
            "Check that encoder modules expose .name with 'blocks_*' and 'head_swish'."
        )

    return feats


def _select_pyramid(feats_by_hw: Dict[Tuple[int, int], torch.Tensor], n_skips: int = 4):
    """
    Build pyramid:
      bottleneck = smallest resolution
      skips = next resolutions increasing (small -> large)
    We'll take:
      skips_small_to_large[:n_skips] and bottleneck
    """
    items = list(feats_by_hw.items())
    items.sort(key=lambda kv: kv[0][0] * kv[0][1])  # area asc: smallest first

    bottleneck = items[0][1]
    skips_small_to_large = [t for (_hw, t) in items[1:]]

    if len(skips_small_to_large) < n_skips:
        # still works, but fewer skip steps
        n_skips = len(skips_small_to_large)

    return skips_small_to_large[:n_skips], bottleneck


# ----------------------------
# Efficient-UNet (multiclass/multilabel logits)
# ----------------------------
class EfficientUNet_EfficientNet(nn.Module):
    """
    EfficientNet encoder (your implementation) + UNet decoder.
    Returns logits: (B, C, H, W).

    - Multi-class: use CrossEntropyLoss(logits, target_long)
    - Multi-label: use BCEWithLogitsLoss(logits, target_float)
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        decoder_channels=(512, 256, 128, 64),
        concat_input: bool = True,
        input_fuse_channels: int = 32,
        n_skips: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.concat_input = bool(concat_input)
        self.n_skips = int(n_skips)

        # We will infer skip channels at runtime on first forward (lazy init pattern).
        # But we still need modules: we do it via nn.LazyConv2d blocks for the first conv in each stage.

        self._built = False
        self.decoder_channels = tuple(decoder_channels)
        self.input_fuse_channels = int(input_fuse_channels)

        # placeholders (built on first forward once we know channels)
        self.up_blocks = nn.ModuleList()
        self.up_input = None
        self.seg_head = None

    def _build(self, skips: List[torch.Tensor], bottleneck: torch.Tensor, x_input: torch.Tensor):
        """
        Build decoder based on actual channel dimensions of captured skips.
        """
        # channels
        ch_b = bottleneck.shape[1]
        skip_ch = [s.shape[1] for s in skips]  # small->large

        # up0: bottleneck up + skip0 => in = ch_b + skip_ch[0]
        # out = decoder_channels[0], etc
        in_ch_list = []
        prev_ch = ch_b
        for i in range(len(skips)):
            in_ch_list.append(prev_ch + skip_ch[i])
            prev_ch = self.decoder_channels[i]

        self.up_blocks = nn.ModuleList([
            Up(in_ch=in_ch_list[i], out_ch=self.decoder_channels[i])
            for i in range(len(skips))
        ])

        head_in = self.decoder_channels[len(skips) - 1] if len(skips) > 0 else ch_b

        if self.concat_input:
            # up to input size and concat input (3ch) => in = head_in + x_input_channels
            in_ch = head_in + x_input.shape[1]
            self.up_input = Up(in_ch=in_ch, out_ch=self.input_fuse_channels)
            head_in = self.input_fuse_channels

        self.seg_head = nn.Conv2d(head_in, self.num_classes, kernel_size=1, stride=1, padding=0)

        self._built = True

    def forward(self, x: torch.Tensor):
        x_inp = x

        feats_by_hw = get_efficientnet_skips_by_hw(self.encoder, x)
        skips, bottleneck = _select_pyramid(feats_by_hw, n_skips=self.n_skips)

        if not self._built:
            self._build(skips, bottleneck, x_inp)

            # Force-move newly created decoder parts to the same device/dtype as input
            self.up_blocks = self.up_blocks.to(device=x.device, dtype=x.dtype)
            if self.up_input is not None:
                self.up_input = self.up_input.to(device=x.device, dtype=x.dtype)
            if self.seg_head is not None:
                self.seg_head = self.seg_head.to(device=x.device, dtype=x.dtype)

            self._built = True  # keep explicit

        h = bottleneck
        for i, skip in enumerate(skips):
            h = self.up_blocks[i](h, skip)

        if self.concat_input:
            h = self.up_input(h, x_inp)

        logits = self.seg_head(h)  # (B,C,H,W)
        return logits


# ----------------------------
# Builders like your original API
# ----------------------------
def get_efficientunet_multilabel_b0(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b0", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b1(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b1", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b2(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b2", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b3(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b3", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b4(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b4", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b5(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b5", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b6(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b6", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)

def get_efficientunet_multilabel_b7(num_classes: int, concat_input: bool = True, pretrained: bool = True):
    enc = EfficientNet.encoder("efficientnet-b7", pretrained=pretrained)
    return EfficientUNet_EfficientNet(enc, num_classes=num_classes, concat_input=concat_input)