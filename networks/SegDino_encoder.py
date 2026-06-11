"""
SegDINO_Encoder.py
==================
Encoder basé sur le ViT local (SegDINO_ViT.py).
Aucune dépendance au repo dinov2/dinov3 externe.

Le ViT produit des tokens de patches (B, N, C).
On extrait les features à 4 profondeurs (quartiles des blocs)
et on les reshape en cartes 2D (B, C, h, w).
"""

import torch
import torch.nn as nn

from Ibrahim.networks.SegDino_ViT import build_vit, load_dinov2_weights


class SegDINOEncoder(nn.Module):
    """
    Paramètres
    ----------
    ckpt_path   : chemin vers les poids pré-entraînés DINOv2 (.pth)
                  Si vide ou fichier absent → initialisation aléatoire
    size        : 's' | 'b' | 'l' | 'g'
    img_size    : taille spatiale des images d'entrée (supposée carrée)
    frozen      : si True, tous les paramètres du backbone sont gelés
    out_indices : liste de 4 indices de blocs à extraire
                  None → quartiles automatiques
    """

    def __init__(
        self,
        ckpt_path: str = "",
        size: str = "s",
        img_size: int = 224,
        frozen: bool = True,
        out_indices=None,
    ):
        super().__init__()

        # ── Construire et charger le backbone ViT ──────────────────────────
        self.backbone = build_vit(size=size, img_size=img_size)
        load_dinov2_weights(self.backbone, ckpt_path, verbose=True)

        self.patch_size = self.backbone.patch_size
        self.embed_dim  = self.backbone.embed_dim
        self.img_size   = img_size

        # ── Indices d'extraction (4 profondeurs) ───────────────────────────
        num_blocks = len(self.backbone.blocks)
        if out_indices is None:
            q = num_blocks // 4
            out_indices = [q - 1, 2*q - 1, 3*q - 1, num_blocks - 1]
        self.out_indices = [int(i) for i in out_indices]

        # ── Gel optionnel ──────────────────────────────────────────────────
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------------- #
    @property
    def out_channels(self):
        """Dimensions des canaux de sortie (4 valeurs = embed_dim chacune)."""
        return [self.embed_dim] * 4

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """
        x      : (B, 3, H, W)
        retour : liste de 4 feature maps [(B, C, h, w)] shallow → deep
        """
        B, _, H, W = x.shape
        gh = H // self.patch_size
        gw = W // self.patch_size

        # Patch embedding + CLS token + pos encoding
        tokens  = self.backbone.patch_embed(x)                # (B, N, C)
        cls_tok = self.backbone.cls_token.expand(B, -1, -1)   # (B, 1, C)
        tokens  = torch.cat([cls_tok, tokens], dim=1)          # (B, N+1, C)
        tokens  = tokens + self.backbone.interpolate_pos_encoding(tokens, gh, gw)
        tokens  = self.backbone.pos_drop(tokens)

        # Passer dans les blocs et collecter aux indices voulus
        feats      = []
        target_set = set(self.out_indices)

        for i, blk in enumerate(self.backbone.blocks):
            tokens = blk(tokens)
            if i in target_set:
                patch_tokens = tokens[:, 1:, :]               # drop CLS
                fmap = patch_tokens.transpose(1, 2).reshape(
                    B, self.embed_dim, gh, gw)
                feats.append(fmap)

        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"
        return feats   # [shallow … deep]
