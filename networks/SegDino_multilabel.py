"""
SegDINO_MultiLabel.py
=====================
Modèle complet = SegDINOEncoder (ViT local) + SegDINODecoder (UNet local).

Aucune dépendance au repo dinov2/dinov3.
Drop-in replacement de EfficientUNet dans le pipeline Synapse.

Usage
-----
    from Ibrahim.networks.SegDINO_MultiLabel import get_segdino_model

    model = get_segdino_model(
        num_classes      = 9,
        ckpt_path        = "./web_pth/dinov3_vits16_pretrain.pth",
        size             = "s",
        img_size         = 224,
        frozen_encoder   = True,
        decoder_channels = [256, 128, 64, 32],
    )
    logits = model(x)   # x: (B,3,H,W) → logits: (B,9,H,W)
"""

import torch
import torch.nn as nn

from Ibrahim.networks.SegDino_encoder import SegDINOEncoder
from Ibrahim.networks.SegDino_decoder import SegDINODecoder


# =========================================================================== #
#  Modèle complet                                                              #
# =========================================================================== #

class SegDINO(nn.Module):
    """
    Paramètres
    ----------
    num_classes      : int   — classes de segmentation (background inclus)
    ckpt_path        : str   — poids pré-entraînés DINOv2 (.pth)
    size             : str   — 's' | 'b' | 'l' | 'g'
    img_size         : int   — taille spatiale H=W de l'entrée
    frozen_encoder   : bool  — geler l'encoder
    decoder_channels : list  — 4 ints, canaux des UpBlocks du décodeur
    out_indices      : list  — 4 ints, indices des blocs ViT à extraire
    """

    def __init__(
        self,
        num_classes: int = 9,
        ckpt_path: str = "",
        size: str = "s",
        img_size: int = 224,
        frozen_encoder: bool = True,
        decoder_channels=None,
        out_indices=None,
    ):
        super().__init__()

        self.encoder = SegDINOEncoder(
            ckpt_path   = ckpt_path,
            size        = size,
            img_size    = img_size,
            frozen      = frozen_encoder,
            out_indices = out_indices,
        )

        self.decoder = SegDINODecoder(
            encoder_channels = self.encoder.out_channels,
            decoder_channels = decoder_channels,
            num_classes      = num_classes,
            img_size         = img_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.encoder(x)
        logits = self.decoder(feats)
        return logits


# =========================================================================== #
#  Factory                                                                     #
# =========================================================================== #

def get_segdino_model(
    num_classes: int = 9,
    ckpt_path: str = "",
    size: str = "s",
    img_size: int = 224,
    frozen_encoder: bool = True,
    decoder_channels=None,
    out_indices=None,
) -> SegDINO:
    """Construit et retourne un modèle SegDINO."""
    return SegDINO(
        num_classes      = num_classes,
        ckpt_path        = ckpt_path,
        size             = size,
        img_size         = img_size,
        frozen_encoder   = frozen_encoder,
        decoder_channels = decoder_channels,
        out_indices      = out_indices,
    )
