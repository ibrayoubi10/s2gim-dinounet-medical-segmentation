"""
SegDINO_ViT.py
==============
Vision Transformer (ViT) implémenté from scratch, compatible avec les
poids pré-entraînés DINOv2 / DINOv3 de Facebook Research.

Pas de dépendance externe au repo dinov2/dinov3 — tout est ici.

Supporte : vit_small, vit_base, vit_large, vit_giant
Patch sizes : 14 ou 16

Chargement des poids
--------------------
Les checkpoints DINOv2 officiels ont le format :
    { "model": { "backbone.xxx": tensor, ... } }
ou directement :
    { "xxx": tensor, ... }

La fonction `load_dinov2_weights()` gère les deux cas et filtre
automatiquement les clés incompatibles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  Helpers                                                                     #
# =========================================================================== #

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


# =========================================================================== #
#  Multi-Head Self-Attention                                                   #
# =========================================================================== #

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =========================================================================== #
#  MLP                                                                         #
# =========================================================================== #

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# =========================================================================== #
#  Transformer Block                                                           #
# =========================================================================== #

class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0, drop_path=0.0,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        init_values=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.ls1   = LayerScale(dim, init_values=init_values)
        self.dp1   = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2   = LayerScale(dim, init_values=init_values)
        self.dp2   = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x


# =========================================================================== #
#  Patch Embedding                                                             #
# =========================================================================== #

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, embed_dim)
        x = self.proj(x)                        # (B, E, gh, gw)
        x = x.flatten(2).transpose(1, 2)        # (B, N, E)
        return x


# =========================================================================== #
#  Vision Transformer                                                          #
# =========================================================================== #

class VisionTransformer(nn.Module):
    """
    ViT compatible avec les poids DINOv2.

    Paramètres principaux
    ---------------------
    img_size    : int
    patch_size  : int (16 pour vit_s/b/l, 14 pour vit_g)
    embed_dim   : int
    depth       : int   (nombre de blocs)
    num_heads   : int
    mlp_ratio   : float (4.0 par défaut)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=1.0,
        num_classes=0,          # 0 = pas de head de classification
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(p=drop_rate)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # optional classification head (num_classes=0 → identity)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    # ---------------------------------------------------------------------- #
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #
    def interpolate_pos_encoding(self, x, gh, gw):
        """Resize pos embedding to match current (gh, gw) grid."""
        N0 = self.pos_embed.shape[1] - 1          # original num patches
        if N0 == gh * gw:
            return self.pos_embed
        cls_pe   = self.pos_embed[:, :1, :]
        patch_pe = self.pos_embed[:, 1:, :]       # (1, N0, C)
        gh0 = gw0 = int(math.sqrt(N0))
        C   = patch_pe.shape[2]
        patch_pe = patch_pe.reshape(1, gh0, gw0, C).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(gh, gw),
                                 mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gh * gw, C)
        return torch.cat([cls_pe, patch_pe], dim=1)

    # ---------------------------------------------------------------------- #
    def forward(self, x):
        """Standard forward — returns CLS token logits (for classification)."""
        B, _, H, W = x.shape
        gh = H // self.patch_size
        gw = W // self.patch_size

        x        = self.patch_embed(x)
        cls_tok  = self.cls_token.expand(B, -1, -1)
        x        = torch.cat([cls_tok, x], dim=1)
        x        = x + self.interpolate_pos_encoding(x, gh, gw)
        x        = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    # ---------------------------------------------------------------------- #
    def get_intermediate_layers(self, x, n=4):
        """
        Retourne les n derniers blocs de feature maps patch tokens.

        Retourne
        --------
        list[Tensor]  : longueur n, chaque (B, N, C)
        """
        B, _, H, W = x.shape
        gh = H // self.patch_size
        gw = W // self.patch_size

        x        = self.patch_embed(x)
        cls_tok  = self.cls_token.expand(B, -1, -1)
        x        = torch.cat([cls_tok, x], dim=1)
        x        = x + self.interpolate_pos_encoding(x, gh, gw)
        x        = self.pos_drop(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= len(self.blocks) - n:
                outputs.append(x[:, 1:, :])   # patch tokens only, drop CLS
        return outputs  # list of (B, N, C)


# =========================================================================== #
#  Configs prédéfinies                                                         #
# =========================================================================== #

_CONFIGS = {
    "vit_small": dict(patch_size=16, embed_dim=384,  depth=12, num_heads=6),
    "vit_base":  dict(patch_size=16, embed_dim=768,  depth=12, num_heads=12),
    "vit_large": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    "vit_giant": dict(patch_size=14, embed_dim=1536, depth=40, num_heads=24),
}

_SIZE_TO_ARCH = {
    "s": "vit_small",
    "b": "vit_base",
    "l": "vit_large",
    "g": "vit_giant",
}


def build_vit(size="s", img_size=224, **kwargs) -> VisionTransformer:
    """
    Construit un ViT DINOv2 par taille ('s', 'b', 'l', 'g').

    Paramètres supplémentaires passés à VisionTransformer (ex: drop_path_rate).
    """
    arch = _SIZE_TO_ARCH[size]
    cfg  = dict(_CONFIGS[arch])
    cfg.update(kwargs)
    return VisionTransformer(img_size=img_size, **cfg)


# =========================================================================== #
#  Chargement des poids DINOv2                                                 #
# =========================================================================== #

def load_dinov2_weights(model: VisionTransformer, ckpt_path: str, verbose=True):
    """
    Charge les poids d'un checkpoint DINOv2/DINOv3 dans le modèle.

    Gère les formats :
        - {"model": {...}}
        - {"teacher": {"backbone.xxx": ...}}
        - {"state_dict": {...}}
        - directement {clé: tensor}

    Préfixes nettoyés automatiquement :
        module. / backbone. / encoder. / model.
    """
    import os
    if not os.path.isfile(ckpt_path):
        if verbose:
            print(f"[SegDINO] WARNING: ckpt not found at '{ckpt_path}' — random init")
        return model

    state = torch.load(ckpt_path, map_location="cpu")

    # Unwrap nested dicts
    for key in ("model", "teacher", "state_dict", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    # Strip common prefixes
    prefixes = ("module.", "backbone.", "encoder.", "model.")
    cleaned = {}
    for k, v in state.items():
        for pfx in prefixes:
            if k.startswith(pfx):
                k = k[len(pfx):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if verbose:
        print(f"[SegDINO] weights loaded from '{ckpt_path}'")
        print(f"          missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"          missing keys (first 5): {missing[:5]}")
    return model
