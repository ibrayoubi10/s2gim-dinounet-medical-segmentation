# HiFormer_configs.py
#
# Drop-in replacement for your config file:
# - No wget dependency
# - No relative ./weights (uses a writable absolute folder)
# - Downloads Swin Tiny pretrained once (atomic .tmp -> final)
#
# Optional:
#   export HIFORMER_WEIGHTS_DIR=/some/writable/path/hiformer_weights

import os
import ml_collections
import urllib.request

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Where to store pretrained weights (writable path)
WEIGHTS_DIR = os.environ.get("HIFORMER_WEIGHTS_DIR", os.path.join(_THIS_DIR, "weights"))
os.makedirs(WEIGHTS_DIR, exist_ok=True)

SWIN_TINY_NAME = "swin_tiny_patch4_window7_224.pth"
SWIN_TINY_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
    "swin_tiny_patch4_window7_224.pth"
)


def _download(url: str, dst: str):
    """
    Robust download to dst using a temporary file in the same directory.
    Avoids permission issues from wget/tempfile(dir='.')
    """
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    tmp = dst + ".tmp"
    print(f"Downloading Swin-transformer model -> {dst}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _ensure_swin_tiny(cfg: ml_collections.ConfigDict):
    dst = os.path.join(WEIGHTS_DIR, SWIN_TINY_NAME)
    _download(SWIN_TINY_URL, dst)
    cfg.swin_pretrained_path = dst
    return cfg


# =========================
# HiFormer-S Configs
# =========================
def get_hiformer_s_configs(num_classes: int = 9, image_size: int = 224):
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = int(image_size)
    cfg.patch_size = 4
    cfg.num_classes = int(num_classes)
    _ensure_swin_tiny(cfg)

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio = (1.0, 1.0, 1.0)
    cfg.drop_rate = 0.0
    cfg.attn_drop_rate = 0.0
    cfg.drop_path_rate = 0.0
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# =========================
# HiFormer-B Configs
# =========================
def get_hiformer_b_configs(num_classes: int = 9, image_size: int = 224):
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = int(image_size)
    cfg.patch_size = 4
    cfg.num_classes = int(num_classes)
    _ensure_swin_tiny(cfg)

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm = [256, 512, 1024]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio = (2.0, 2.0, 1.0)
    cfg.drop_rate = 0.0
    cfg.attn_drop_rate = 0.0
    cfg.drop_path_rate = 0.0
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# =========================
# HiFormer-L Configs
# =========================
def get_hiformer_l_configs(num_classes: int = 9, image_size: int = 224):
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = int(image_size)
    cfg.patch_size = 4
    cfg.num_classes = int(num_classes)
    _ensure_swin_tiny(cfg)

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 4, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio = (4.0, 4.0, 1.0)
    cfg.drop_rate = 0.0
    cfg.attn_drop_rate = 0.0
    cfg.drop_path_rate = 0.0
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Convenience helper if you want one function
def get_hiformer_configs(variant: str = "S", num_classes: int = 9, image_size: int = 224):
    v = str(variant).lower().strip()
    if v in ("s", "small"):
        return get_hiformer_s_configs(num_classes=num_classes, image_size=image_size)
    if v in ("b", "base"):
        return get_hiformer_b_configs(num_classes=num_classes, image_size=image_size)
    if v in ("l", "large"):
        return get_hiformer_l_configs(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Unknown HiFormer variant: {variant} (use S/B/L)")