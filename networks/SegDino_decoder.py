"""
SegDINO_Decoder.py
==================
Décodeur UNet avec skip connections, entièrement local.
Reçoit 4 feature maps de l'encoder (shallow → deep)
et reconstruit la segmentation à la résolution d'entrée.

Architecture
------------
    f4 (deep, petite résolution)
        │
    Bottleneck ──────────────────────────────────────────────────
        │ Upsample ×2                                            │
        + skip f3 → DoubleConv                                   │
          │ Upsample ×2                                          │
          + skip f2 → DoubleConv                                 │
            │ Upsample ×2                                        │
            + skip f1 → DoubleConv                               │
              │ Upsample ×2 (→ résolution pleine image)          │
              DoubleConv → Conv1×1 → logits (B, C_cls, H, W) ───┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  Briques de base                                                             #
# =========================================================================== #

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample ×2 (bilinear) → concat skip → DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =========================================================================== #
#  Décodeur                                                                    #
# =========================================================================== #

class SegDINODecoder(nn.Module):
    """
    Paramètres
    ----------
    encoder_channels : list[int] longueur 4 — canaux de l'encoder (shallow→deep)
                       Pour ViT-S/B/L/G : typiquement [384]*4, [768]*4, etc.
    decoder_channels : list[int] longueur 4 — canaux de sortie de chaque UpBlock
    num_classes      : int — nombre de classes de segmentation
    img_size         : int — taille finale H=W de la sortie
    """

    def __init__(
        self,
        encoder_channels,
        decoder_channels=None,
        num_classes: int = 9,
        img_size: int = 224,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
        assert len(encoder_channels) == 4
        assert len(decoder_channels) == 4

        ec = encoder_channels   # [e1, e2, e3, e4]  shallow → deep
        dc = decoder_channels   # [d1, d2, d3, d4]

        # Bottleneck sur la feature la plus profonde
        self.bottleneck = DoubleConv(ec[3], dc[0])

        # UpBlocks avec skips
        self.up3 = UpBlock(dc[0], ec[2], dc[1])   # concat avec f3
        self.up2 = UpBlock(dc[1], ec[1], dc[2])   # concat avec f2
        self.up1 = UpBlock(dc[2], ec[0], dc[3])   # concat avec f1

        # Dernier up sans skip (patch→image resolution)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(dc[3], dc[3]),
        )

        self.seg_head = nn.Conv2d(dc[3], num_classes, kernel_size=1)
        self.img_size  = img_size

    # ---------------------------------------------------------------------- #
    def forward(self, feats):
        """
        feats : list[Tensor] longueur 4, shallow → deep
                chaque (B, C, h, w)

        retour : logits (B, num_classes, H, W)
        """
        f1, f2, f3, f4 = feats

        x = self.bottleneck(f4)   # (B, dc[0], h4, w4)
        x = self.up3(x, f3)       # (B, dc[1], h3, w3)
        x = self.up2(x, f2)       # (B, dc[2], h2, w2)
        x = self.up1(x, f1)       # (B, dc[3], h1, w1)
        x = self.up0(x)           # (B, dc[3], 2·h1, 2·w1)

        logits = self.seg_head(x)

        # Ajuster à img_size si nécessaire
        if logits.shape[2] != self.img_size or logits.shape[3] != self.img_size:
            logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits
