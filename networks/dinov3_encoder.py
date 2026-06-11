import math
from typing import Any, Optional, List, Dict

import torch
import torch.nn as nn
import timm


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str, strict: bool = False):
    """
    Flexible checkpoint loader for local .pth checkpoints.
    Handles several common checkpoint formats.
    """
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format: checkpoint is not a dict.")

    candidate_keys = ["state_dict", "model", "teacher", "student", "module", "network"]
    state_dict = None

    # case 1: checkpoint is directly a state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        for k in candidate_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break

    if state_dict is None:
        raise RuntimeError(f"Could not find usable state_dict. Keys found: {list(ckpt.keys())}")

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k

        changed = True
        while changed:
            changed = False
            for prefix in ["module.", "model.", "backbone.", "teacher.", "student."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    changed = True

        cleaned[new_k] = v

    incompatible = model.load_state_dict(cleaned, strict=strict)

    if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
    else:
        missing, unexpected = incompatible

    print(f"[INFO] strict={strict}")
    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[INFO] sample missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[INFO] sample unexpected keys:", unexpected[:20])

    return missing, unexpected


class DINOv3Backbone(nn.Module):
    """
    DINOv3 ViT backbone for segmentation.

    Main idea:
    - Use timm ViT backbone
    - Extract intermediate transformer block outputs
    - Convert token sequences (B, N, C) into spatial maps (B, C, H, W)

    Output modes:
    - return_multiscale=False:
        returns one feature map: (B, C, Ht, Wt)

    - return_multiscale=True:
        returns a dict:
        {
            "stage1": f1,
            "stage2": f2,
            "stage3": f3,
            "stage4": f4,
        }

    Notes:
    - For ViT-B/16 with input 224x224:
        patch grid is typically 14x14
    - Since ViT is not hierarchical by design, all extracted stages have the
      same spatial resolution, but come from different depths.
    - This is still much better than using only the final block.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        weights_path: Optional[str] = None,
        pretrained: bool = False,
        freeze: bool = False,
        remove_cls_token: bool = True,
        return_multiscale: bool = True,
        out_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.weights_path = weights_path
        self.pretrained = pretrained
        self.freeze = freeze
        self.remove_cls_token = remove_cls_token
        self.return_multiscale = return_multiscale
        self.verbose = verbose

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if weights_path is not None and len(weights_path) > 0:
            load_checkpoint_flexible(self.backbone, weights_path, strict=False)

        self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Could not infer backbone.num_features")

        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)
        self.patch_embed = getattr(self.backbone, "patch_embed", None)
        self.blocks = getattr(self.backbone, "blocks", None)
        self.norm = getattr(self.backbone, "norm", None)

        if self.patch_embed is None or self.blocks is None:
            raise RuntimeError("This implementation expects a ViT-like timm model with patch_embed and blocks.")

        num_blocks = len(self.blocks)

        if out_indices is None:
            # choose 4 roughly spread-out blocks
            if num_blocks >= 12:
                out_indices = [2, 5, 8, 11]
            elif num_blocks >= 8:
                out_indices = [1, 3, 5, 7]
            elif num_blocks >= 4:
                out_indices = [0, 1, 2, 3]
            else:
                raise RuntimeError(f"Backbone has too few transformer blocks: {num_blocks}")

        if len(out_indices) != 4:
            raise ValueError("out_indices must contain exactly 4 block indices.")

        self.out_indices = out_indices

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if self.verbose:
            print(f"[INFO] model_name         : {self.model_name}")
            print(f"[INFO] embed_dim          : {self.embed_dim}")
            print(f"[INFO] num_prefix_tokens  : {self.num_prefix_tokens}")
            print(f"[INFO] num_blocks         : {num_blocks}")
            print(f"[INFO] out_indices        : {self.out_indices}")
            print(f"[INFO] return_multiscale  : {self.return_multiscale}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_tensor_from_any(self, x: Any, name: str = "object") -> torch.Tensor:
        """
        Robustly extract a tensor from:
        - Tensor
        - tuple/list containing tensors
        - dict containing tensors
        """
        if isinstance(x, torch.Tensor):
            return x

        if isinstance(x, (tuple, list)):
            tensor_vals = [v for v in x if isinstance(v, torch.Tensor)]
            if len(tensor_vals) == 0:
                raise RuntimeError(f"{name} is tuple/list but contains no tensor.")
            if self.verbose:
                print(f"[DEBUG] {name} was {type(x).__name__}, using tensor with shape {tuple(tensor_vals[0].shape)}")
            return tensor_vals[0]

        if isinstance(x, dict):
            tensor_items = [(k, v) for k, v in x.items() if isinstance(v, torch.Tensor)]
            if len(tensor_items) == 0:
                raise RuntimeError(f"{name} is dict but contains no tensor.")
            k, v = tensor_items[0]
            if self.verbose:
                print(f"[DEBUG] {name} was dict, using key '{k}' with shape {tuple(v.shape)}")
            return v

        raise RuntimeError(f"Unsupported {name} type: {type(x)}")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _get_patch_grid_hw(self, x: torch.Tensor):
        """
        Infer patch grid size after patch embedding.
        More robust across timm ViT / EVA variants.
        """
        B, C, H, W = x.shape

        if hasattr(self.patch_embed, "grid_size") and self.patch_embed.grid_size is not None:
            grid_size = self.patch_embed.grid_size
            if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
                return int(grid_size[0]), int(grid_size[1])

        patch_size = getattr(self.patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph = pw = patch_size

        gh = H // ph
        gw = W // pw
        return gh, gw

    def _extract_tensor_from_forward_features(self, feats: Any) -> torch.Tensor:
        """
        Handle common timm forward_features outputs.
        """
        if isinstance(feats, dict):
            preferred_keys = [
                "x_norm_patchtokens",
                "patch_tokens",
                "x_prenorm",
                "tokens",
                "features",
                "last_hidden_state",
            ]
            for key in preferred_keys:
                if key in feats and isinstance(feats[key], torch.Tensor):
                    if self.verbose:
                        print(f"[DEBUG] using dict key: {key} with shape {tuple(feats[key].shape)}")
                    return feats[key]

            tensor_items = [(k, v) for k, v in feats.items() if isinstance(v, torch.Tensor)]
            if len(tensor_items) == 0:
                raise RuntimeError(f"Unsupported dict output keys: {list(feats.keys())}")

            k, v = tensor_items[0]
            if self.verbose:
                print(f"[DEBUG] fallback dict key: {k} with shape {tuple(v.shape)}")
            return v

        if isinstance(feats, (tuple, list)):
            tensor_vals = [v for v in feats if isinstance(v, torch.Tensor)]
            if len(tensor_vals) == 0:
                raise RuntimeError("Unsupported tuple/list output from forward_features")
            if self.verbose:
                print(f"[DEBUG] using tuple/list tensor with shape {tuple(tensor_vals[-1].shape)}")
            return tensor_vals[-1]

        if isinstance(feats, torch.Tensor):
            if self.verbose:
                print(f"[DEBUG] forward_features returned tensor shape {tuple(feats.shape)}")
            return feats

        raise RuntimeError(f"Unsupported forward_features output type: {type(feats)}")

    def _tokens_to_map(self, feats: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """
        Convert tokens into spatial map.

        Supports:
        - (B, C, H, W): already spatial
        - (B, N, C): token sequence
        """
        if feats.ndim == 4:
            return feats

        if feats.ndim != 3:
            raise RuntimeError(f"Unsupported feature tensor shape: {tuple(feats.shape)}")

        B, N, C = feats.shape

        # Case 1: pure patch tokens
        if N == gh * gw:
            return feats.transpose(1, 2).contiguous().view(B, C, gh, gw)

        # Case 2: prefixed tokens
        num_prefix = self.num_prefix_tokens if self.num_prefix_tokens is not None else 1

        if self.remove_cls_token:
            if N == num_prefix + gh * gw:
                feats = feats[:, num_prefix:, :]
                return feats.transpose(1, 2).contiguous().view(B, C, gh, gw)

            # fallback candidates
            for prefix in [1, 5]:
                if N == prefix + gh * gw:
                    feats = feats[:, prefix:, :]
                    return feats.transpose(1, 2).contiguous().view(B, C, gh, gw)

        # Case 3: infer square map directly from token count
        if self.remove_cls_token:
            for prefix in [num_prefix, 1, 5]:
                if N > prefix:
                    n_patch = N - prefix
                    side = int(math.sqrt(n_patch))
                    if side * side == n_patch:
                        feats = feats[:, prefix:, :]
                        return feats.transpose(1, 2).contiguous().view(B, C, side, side)

        # Case 4: no prefix, square tokens directly
        side = int(math.sqrt(N))
        if side * side == N:
            return feats.transpose(1, 2).contiguous().view(B, C, side, side)

        raise RuntimeError(
            f"Cannot reshape tokens into feature map. Got N={N}, "
            f"gh={gh}, gw={gw}, num_prefix_tokens={num_prefix}."
        )
    # ------------------------------------------------------------------
    # Raw forward modes
    # ------------------------------------------------------------------
    def forward_features_raw(self, x: torch.Tensor) -> Any:
        """
        Raw timm backbone forward_features output.
        """
        return self.backbone.forward_features(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw tensor extracted from forward_features.
        Usually:
            - (B, N, C) for ViT tokens
            - or (B, C, H, W) for spatial output
        """
        raw = self.backbone.forward_features(x)

        if self.verbose:
            print(f"[DEBUG] raw output type: {type(raw)}")
            if isinstance(raw, dict):
                print(f"[DEBUG] raw dict keys: {list(raw.keys())}")

        feats = self._extract_tensor_from_forward_features(raw)

        if self.verbose and isinstance(feats, torch.Tensor):
            print(f"[DEBUG] extracted feats shape: {tuple(feats.shape)}")

        return feats

    # ------------------------------------------------------------------
    # Intermediate block extraction
    # ------------------------------------------------------------------
    def _forward_intermediates(self, x: torch.Tensor):
        """
        Extract 4 intermediate token tensors from selected transformer blocks.

        Works with timm EVA / DINOv3 variants where:
        - patch_embed(x) may return Tensor / tuple / list
        - _pos_embed may expect 4D input
        - output may be BCHW, BHWC, or BNC

        Returns:
            outs: list of 4 tensors, each typically (B, N, C)
            gh, gw: TRUE patch grid size inferred from actual patch embedding output
        """
        # patch embedding
        x_tokens = self.patch_embed(x)
        x_tokens = self._extract_tensor_from_any(x_tokens, name="patch_embed output")

        # IMPORTANT:
        # infer TRUE patch grid from actual patch_embed output, not from input size
        gh = gw = None

        if x_tokens.ndim == 4:
            if x_tokens.shape[1] == self.embed_dim:
                # BCHW
                B, C, Ht, Wt = x_tokens.shape
                gh, gw = Ht, Wt
            else:
                # assume BHWC
                B, Ht, Wt, C = x_tokens.shape
                gh, gw = Ht, Wt

        # EVA-style _pos_embed may expect 4D input, so do this before flattening.
        if hasattr(self.backbone, "_pos_embed"):
            x_tokens = self.backbone._pos_embed(x_tokens)
            x_tokens = self._extract_tensor_from_any(x_tokens, name="_pos_embed output")
        elif hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            x_tokens = self._extract_tensor_from_any(x_tokens, name="pre-pos_embed tensor")

            if x_tokens.ndim == 4:
                if x_tokens.shape[1] == self.embed_dim:
                    # BCHW -> BNC
                    B, C, Ht, Wt = x_tokens.shape
                    gh, gw = Ht, Wt
                    x_tokens = x_tokens.flatten(2).transpose(1, 2).contiguous()
                else:
                    # BHWC -> BNC
                    B, Ht, Wt, C = x_tokens.shape
                    gh, gw = Ht, Wt
                    x_tokens = x_tokens.reshape(B, Ht * Wt, C).contiguous()

            pos_embed = self.backbone.pos_embed
            if pos_embed.ndim == 3 and pos_embed.shape[1] == x_tokens.shape[1]:
                x_tokens = x_tokens + pos_embed

        # Ensure tokens are (B, N, C)
        x_tokens = self._extract_tensor_from_any(x_tokens, name="tokens before blocks")

        if x_tokens.ndim == 4:
            if x_tokens.shape[1] == self.embed_dim:
                # BCHW -> BNC
                B, C, Ht, Wt = x_tokens.shape
                gh, gw = Ht, Wt
                x_tokens = x_tokens.flatten(2).transpose(1, 2).contiguous()
            else:
                # assume BHWC -> BNC
                B, Ht, Wt, C = x_tokens.shape
                gh, gw = Ht, Wt
                x_tokens = x_tokens.reshape(B, Ht * Wt, C).contiguous()

        if x_tokens.ndim != 3:
            raise RuntimeError(f"Expected tokens to be 3D or 4D before blocks, got shape {tuple(x_tokens.shape)}")

        # If gh/gw still unknown, infer from token count
        if gh is None or gw is None:
            B, N, C = x_tokens.shape
            num_prefix = self.num_prefix_tokens if self.num_prefix_tokens is not None else 1

            if N > num_prefix:
                n_patch = N - num_prefix
                side = int(math.sqrt(n_patch))
                if side * side == n_patch:
                    gh = gw = side
                else:
                    side = int(math.sqrt(N))
                    if side * side == N:
                        gh = gw = side
                    else:
                        raise RuntimeError(
                            f"Could not infer patch grid from token count N={N}. "
                            f"num_prefix_tokens={num_prefix}"
                        )
            else:
                raise RuntimeError(f"Invalid token count N={N} for prefix removal.")

        if hasattr(self.backbone, "patch_drop") and self.backbone.patch_drop is not None:
            x_tokens = self.backbone.patch_drop(x_tokens)
            x_tokens = self._extract_tensor_from_any(x_tokens, name="patch_drop output")

        if hasattr(self.backbone, "norm_pre") and self.backbone.norm_pre is not None:
            x_tokens = self.backbone.norm_pre(x_tokens)
            x_tokens = self._extract_tensor_from_any(x_tokens, name="norm_pre output")

        outs = []
        for i, blk in enumerate(self.blocks):
            x_tokens = blk(x_tokens)
            x_tokens = self._extract_tensor_from_any(x_tokens, name=f"block {i} output")
            if i in self.out_indices:
                outs.append(x_tokens)

        if len(outs) != 4:
            raise RuntimeError(f"Expected 4 intermediate outputs, got {len(outs)}")

        if self.norm is not None:
            outs[-1] = self.norm(outs[-1])
            outs[-1] = self._extract_tensor_from_any(outs[-1], name="final norm output")

        if self.verbose:
            print(f"[DEBUG] TRUE patch grid: gh={gh}, gw={gw}")
            for idx, o in zip(self.out_indices, outs):
                print(f"[DEBUG] block {idx} token shape: {tuple(o.shape)}")

        return outs, gh, gw

    def forward_intermediates(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return 4 spatial feature maps from selected transformer depths.

        Output:
        {
            "stage1": (B, C, gh, gw),
            "stage2": (B, C, gh, gw),
            "stage3": (B, C, gh, gw),
            "stage4": (B, C, gh, gw),
        }
        """
        outs, gh, gw = self._forward_intermediates(x)

        fmaps = [self._tokens_to_map(t, gh, gw) for t in outs]

        out_dict = {
            "stage1": fmaps[0],
            "stage2": fmaps[1],
            "stage3": fmaps[2],
            "stage4": fmaps[3],
        }

        if self.verbose:
            for k, v in out_dict.items():
                print(f"[DEBUG] {k} shape: {tuple(v.shape)}")

        return out_dict

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Default forward.

        If return_multiscale=True:
            returns dict with 4 stages

        Else:
            returns one spatial feature map (final stage)
        """
        if self.return_multiscale:
            return self.forward_intermediates(x)

        feats = self.forward_tokens(x)
        gh, gw = self._get_patch_grid_hw(x)
        fmap = self._tokens_to_map(feats, gh, gw)

        if self.verbose:
            print(f"[DEBUG] single feature map shape: {tuple(fmap.shape)}")

        return fmap


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    model = DINOv3Backbone(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        weights_path="/data3/nkozah/my_project/Ibrahim_Dino_Unet/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        pretrained=False,
        freeze=False,
        remove_cls_token=True,
        return_multiscale=True,
        out_indices=[2, 5, 8, 11],
        verbose=True,
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 224, 224).to(device)
    print(f"[INFO] input shape: {tuple(x.shape)}")

    with torch.no_grad():
        feats = model(x)

    print("\n[RESULTS]")
    if isinstance(feats, dict):
        for k, v in feats.items():
            print(k, v.shape)
    else:
        print("output shape:", feats.shape)