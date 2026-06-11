"""
https://github.com/amirhossein-kz/HiFormer/blob/main/models/HiFormer.py
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from Ibrahim.networks.HiFormer_Encoder import All2Cross
from Ibrahim.networks.HiFormer_Decoder import ConvUpsample, SegmentationHead, ReconstructionHead, SuperpixelPooling
import torch.nn.functional as F

class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128 ,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
        )

        self.reconstruction_head = ReconstructionHead(
            in_channels=16,
            out_channels=3,
            kernel_size=1,
        )

        self.SP = SuperpixelPooling

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x, train=False, superpixel_map=None):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        # out = self.segmentation_head(C)  # multi-class classification
        seg_out = F.sigmoid(self.segmentation_head(C).squeeze(1))  # binary classification

        if train==False:
            return seg_out
        else:
            recontrust_out = self.reconstruction_head(C)
            SPAttention_fea_batch = self.SP(C, superpixel_map)  # list(32*[n, 256])
            SurPLocalCls_batch = []
            for sp in range(len(SPAttention_fea_batch)):
                x_locals_out_sp = []
                for tk in range(SPAttention_fea_batch[sp].shape[0]):
                    # cls_out = self.cls_head(SPAttention_fea_batch[sp][tk])
                    # x_locals_out_sp.append(cls_out)
                    x_locals_out_sp.append(SPAttention_fea_batch[sp][tk])

                x_locals_out_sp = torch.stack(x_locals_out_sp)
                SurPLocalCls_batch.extend(x_locals_out_sp)
            SurPLocalCls_batch = torch.stack(SurPLocalCls_batch)
            # """output: seg_pred; reconstruction out; local preds for superpixels; local feas for superpixels"""
            # return seg_out, recontrust_out, SurPLocalCls_batch, SPAttention_fea_batch
            """output: seg_pred; reconstruction out; local preds for superpixels"""
            return seg_out, recontrust_out, SurPLocalCls_batch


"""
HiFormer_MultiLabel for Synapse (multi-class / multi-label-ready head)

- Synapse is typically multi-class segmentation (C=9 organs + background depending on your setup).
- This implementation outputs:
    * logits:  (B, C, H, W)
    * probs:   softmax(logits) (optional)
- Keeps your train-time extra heads (reconstruction + superpixel pooling) exactly like your base code.

Notes:
- For Synapse you likely have in_chans=1 (CT slice). Set in_chans accordingly.
- DO NOT squeeze channel dim.
- Use CrossEntropyLoss for multi-class (single label per pixel).
  If you truly want multi-label per pixel, use BCEWithLogitsLoss and sigmoid instead.
"""

class HiFormer_MultiLabel(nn.Module):
    def __init__(
        self,
        config,
        img_size=224,
        in_chans=1,
        n_classes=9,
        patch_size=(4, 16),
        return_probs=False,
        multilabel=False,
    ):
        """
        Args:
            config: HiFormer encoder config
            img_size: input size (must match your pipeline resize/crop)
            in_chans: 1 for Synapse CT, 3 if RGB
            n_classes: number of classes (e.g., 9 for Synapse organs)
            patch_size: (small, large) patch sizes used by encoder tokens
            return_probs: if True, also returns probabilities (softmax/sigmoid)
            multilabel: if True -> sigmoid (multi-label per pixel).
                        if False -> softmax (multi-class per pixel).
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = list(patch_size)
        self.n_classes = n_classes
        self.return_probs = return_probs
        self.multilabel = multilabel

        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)

        # Encoder outputs: your original code assumes:
        # - i==0 -> embed dim 96 (large tokens) -> ConvUp_l (no upsample)
        # - i==1 -> embed dim 384 (small tokens) -> ConvUp_s (upsample)
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
        )

        # Optional reconstruction head (kept from your code)
        # If in_chans=1 you probably want out_channels=1 reconstruction
        self.reconstruction_head = ReconstructionHead(
            in_channels=16,
            out_channels=in_chans,
            kernel_size=1,
        )

        self.SP = SuperpixelPooling

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

    def _tokens_to_map(self, token_tensor: torch.Tensor, i: int) -> torch.Tensor:
        """
        token_tensor: (B, 1 + HW, D) or (B, HW, D) depending on All2Cross
        We expect cls token exists -> remove it.
        """
        if token_tensor.dim() != 3:
            raise ValueError(f"Expected token tensor (B, N, D). Got: {token_tensor.shape}")

        # If cls token exists (N = 1 + HW), remove it.
        # Your original code did: x[:, 1:]
        if token_tensor.size(1) > 1:
            tokens = token_tensor[:, 1:]
        else:
            tokens = token_tensor

        h = self.img_size // self.patch_size[i]
        w = self.img_size // self.patch_size[i]
        feat = Rearrange("b (h w) d -> b d h w", h=h, w=w)(tokens)
        return feat

    def forward(self, x, train: bool = False, superpixel_map=None):
        """
        Returns (in eval):
            logits                       if return_probs=False
            (logits, probs)              if return_probs=True

        Returns (in train=True):
            (logits, recon, sp_local)                if return_probs=False
            (logits, probs, recon, sp_local)         if return_probs=True

        Where:
            logits: (B, C, H, W)
            probs : softmax(logits) or sigmoid(logits)
        """
        xs = self.All2Cross(x)  # expected list/tuple of 2 token tensors

        # Convert token embeddings to feature maps, then upsample paths
        reshaped_embed = []
        for i, tok in enumerate(xs):
            feat = self._tokens_to_map(tok, i)
            feat = self.ConvUp_l(feat) if i == 0 else self.ConvUp_s(feat)
            reshaped_embed.append(feat)

        # Fuse
        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        logits = self.segmentation_head(C)  # (B, n_classes, H, W)

        probs = None
        if self.return_probs:
            if self.multilabel:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

        if not train:
            return (logits, probs) if self.return_probs else logits

        # train-time extras (same as your binary version)
        recon = self.reconstruction_head(C)

        if superpixel_map is None:
            raise ValueError("train=True requires superpixel_map (for SuperpixelPooling).")

        sp_feat_list = self.SP(C, superpixel_map)  # list length B: [n_sp, feat_dim] or similar
        sp_local_batch = []
        for sp in range(len(sp_feat_list)):
            x_locals_out_sp = []
            for tk in range(sp_feat_list[sp].shape[0]):
                x_locals_out_sp.append(sp_feat_list[sp][tk])
            x_locals_out_sp = torch.stack(x_locals_out_sp)
            sp_local_batch.extend(x_locals_out_sp)

        sp_local_batch = torch.stack(sp_local_batch)  # (sum_nsp_over_batch, feat_dim)

        if self.return_probs:
            return logits, probs, recon, sp_local_batch
        return logits, recon, sp_local_batch