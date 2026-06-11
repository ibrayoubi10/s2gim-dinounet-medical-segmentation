import torch
import torch.nn as nn
import torch.nn.functional as F

from Ibrahim.networks.dinov3_encoder import DINOv3Backbone


def make_group_norm(num_channels: int, max_groups: int = 8):
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        act=True,
        dropout=0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            make_group_norm(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.conv1 = ConvGNAct(in_ch, out_ch, 3, 1, 1, act=True, dropout=dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, 3, 1, 1, act=False, dropout=0.0)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class FusionBlock(nn.Module):
    """
    Fuse decoder feature with skip feature.
    """
    def __init__(self, dec_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ResidualConvBlock(dec_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, dec_feat, skip_feat):
        if skip_feat.shape[-2:] != dec_feat.shape[-2:]:
            skip_feat = F.interpolate(skip_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([dec_feat, skip_feat], dim=1)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x, target_size=None):
        if target_size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.block(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_ch, num_classes, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, in_ch, 3, 1, 1, act=True, dropout=dropout),
            nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class DINOv3_MultiLabel(nn.Module):
    """
    Robust DINOv3 segmentation model.

    Supports encoder outputs as:
      - Tensor: (B, C, H, W)
      - list/tuple of tensors: multi-scale features
      - dict of tensors: multi-scale features

    Current encoder compatibility:
      - if encoder returns one tensor only, we still build a pseudo pyramid
        from that tensor so the model remains usable.

    Future encoder compatibility:
      - if encoder is modified to return 4 scales, this model will use them directly.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        weights_path: str = None,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        decoder_channels=(512, 256, 128, 64),
        dropout=0.1,
        verbose: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.verbose = verbose
        self.decoder_channels = decoder_channels

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------
        self.encoder = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            freeze=freeze_backbone,
            remove_cls_token=True,
            verbose=verbose,
        )

        enc_ch = self.encoder.embed_dim  # e.g. 768

        # --------------------------------------------------
        # Input feature projections for 4 stages
        # These projections are used after we normalize encoder outputs
        # into 4 feature maps [f1, f2, f3, f4] from shallow -> deep
        # --------------------------------------------------
        self.proj1 = ConvGNAct(enc_ch, decoder_channels[3], kernel_size=1, stride=1, padding=0, act=True)
        self.proj2 = ConvGNAct(enc_ch, decoder_channels[2], kernel_size=1, stride=1, padding=0, act=True)
        self.proj3 = ConvGNAct(enc_ch, decoder_channels[1], kernel_size=1, stride=1, padding=0, act=True)
        self.proj4 = ConvGNAct(enc_ch, decoder_channels[0], kernel_size=1, stride=1, padding=0, act=True)

        # Bottleneck on deepest feature
        self.bottleneck = ResidualConvBlock(decoder_channels[0], decoder_channels[0], dropout=dropout)

        # Decoder path
        self.up4 = UpBlock(decoder_channels[0], decoder_channels[1], dropout=dropout)
        self.fuse3 = FusionBlock(decoder_channels[1], decoder_channels[1], decoder_channels[1], dropout=dropout)

        self.up3 = UpBlock(decoder_channels[1], decoder_channels[2], dropout=dropout)
        self.fuse2 = FusionBlock(decoder_channels[2], decoder_channels[2], decoder_channels[2], dropout=dropout)

        self.up2 = UpBlock(decoder_channels[2], decoder_channels[3], dropout=dropout)
        self.fuse1 = FusionBlock(decoder_channels[3], decoder_channels[3], decoder_channels[3], dropout=dropout)

        # Final refine
        self.refine = nn.Sequential(
            ResidualConvBlock(decoder_channels[3], decoder_channels[3], dropout=dropout),
            ResidualConvBlock(decoder_channels[3], decoder_channels[3], dropout=0.0),
        )

        self.seg_head = SegHead(decoder_channels[3], num_classes, dropout=dropout)

        self._init_decoder_weights()

    # --------------------------------------------------
    # Freeze helpers
    # --------------------------------------------------
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    # --------------------------------------------------
    # Init decoder only
    # --------------------------------------------------
    def _init_decoder_weights(self):
        decoder_modules = [
            self.proj1, self.proj2, self.proj3, self.proj4,
            self.bottleneck,
            self.up4, self.fuse3,
            self.up3, self.fuse2,
            self.up2, self.fuse1,
            self.refine,
            self.seg_head,
        ]

        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GroupNorm):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # --------------------------------------------------
    # Normalize encoder output into 4 feature maps
    # --------------------------------------------------
    def _sort_feature_list(self, feats_list):
        """
        Sort features from shallow -> deep using spatial size.
        Largest HxW first, smallest last.
        """
        feats_list = [f for f in feats_list if isinstance(f, torch.Tensor) and f.ndim == 4]
        feats_list = sorted(feats_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        return feats_list

    def _build_pseudo_pyramid_from_single_map(self, feat):
        """
        Fallback when encoder returns only one map.
        Build 4 pseudo-levels from the same map.

        Suppose feat = (B, C, 14, 14)
        We produce:
          f4 = 14x14   (deep)
          f3 = 28x28
          f2 = 56x56
          f1 = 112x112
        This is not ideal, but keeps the model usable with current encoder.
        """
        f4 = feat
        f3 = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        f2 = F.interpolate(feat, scale_factor=4, mode="bilinear", align_corners=False)
        f1 = F.interpolate(feat, scale_factor=8, mode="bilinear", align_corners=False)
        return [f1, f2, f3, f4]

    def _normalize_encoder_outputs(self, raw_feats):
        """
        Return [f1, f2, f3, f4] from shallow -> deep.
        Each must be (B, C, H, W).
        """
        # Case 1: encoder returns a single feature map
        if isinstance(raw_feats, torch.Tensor):
            if raw_feats.ndim != 4:
                raise RuntimeError(f"Expected encoder tensor output (B,C,H,W), got {tuple(raw_feats.shape)}")
            return self._build_pseudo_pyramid_from_single_map(raw_feats)

        # Case 2: encoder returns list or tuple
        if isinstance(raw_feats, (list, tuple)):
            feats = self._sort_feature_list(list(raw_feats))
            if len(feats) == 0:
                raise RuntimeError("Encoder returned empty feature list/tuple.")
            if len(feats) == 1:
                return self._build_pseudo_pyramid_from_single_map(feats[0])
            if len(feats) >= 4:
                return feats[:4]
            # if 2 or 3 scales only, extend from deepest
            while len(feats) < 4:
                feats.append(feats[-1])
            return feats[:4]

        # Case 3: encoder returns dict
        if isinstance(raw_feats, dict):
            preferred_keys = ["stage1", "stage2", "stage3", "stage4", "f1", "f2", "f3", "f4"]
            chosen = []

            for k in preferred_keys:
                if k in raw_feats and isinstance(raw_feats[k], torch.Tensor) and raw_feats[k].ndim == 4:
                    chosen.append(raw_feats[k])

            if len(chosen) >= 4:
                chosen = self._sort_feature_list(chosen)
                return chosen[:4]

            tensor_vals = [v for v in raw_feats.values() if isinstance(v, torch.Tensor) and v.ndim == 4]
            tensor_vals = self._sort_feature_list(tensor_vals)

            if len(tensor_vals) == 0:
                raise RuntimeError("Encoder dict output contains no 4D tensor features.")

            if len(tensor_vals) == 1:
                return self._build_pseudo_pyramid_from_single_map(tensor_vals[0])

            while len(tensor_vals) < 4:
                tensor_vals.append(tensor_vals[-1])

            return tensor_vals[:4]

        raise RuntimeError(f"Unsupported encoder output type: {type(raw_feats)}")

    # --------------------------------------------------
    # Forward parts
    # --------------------------------------------------
    def forward_features(self, x):
        raw_feats = self.encoder(x)
        feats = self._normalize_encoder_outputs(raw_feats)
        return feats  # [f1, f2, f3, f4]

    def forward_decoder(self, feats):
        """
        feats = [f1, f2, f3, f4] from shallow -> deep
        """
        f1, f2, f3, f4 = feats

        # project channels
        f1 = self.proj1(f1)  # highest spatial, lowest semantic
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)
        f4 = self.proj4(f4)  # lowest spatial, deepest semantic

        # deep bottleneck
        x = self.bottleneck(f4)

        # up + fuse with skip features
        x = self.up4(x, target_size=f3.shape[-2:])
        x = self.fuse3(x, f3)

        x = self.up3(x, target_size=f2.shape[-2:])
        x = self.fuse2(x, f2)

        x = self.up2(x, target_size=f1.shape[-2:])
        x = self.fuse1(x, f1)

        x = self.refine(x)
        return x

    def forward(self, x):
        input_hw = x.shape[-2:]

        feats = self.forward_features(x)
        dec = self.forward_decoder(feats)
        logits = self.seg_head(dec)

        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)

        if self.verbose:
            print(f"[DEBUG] input shape : {tuple(x.shape)}")
            for i, f in enumerate(feats, start=1):
                print(f"[DEBUG] feat{i} shape : {tuple(f.shape)}")
            print(f"[DEBUG] dec shape   : {tuple(dec.shape)}")
            print(f"[DEBUG] logits shape: {tuple(logits.shape)}")

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    model = DINOv3_MultiLabel(
        num_classes=9,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        weights_path="/data3/nkozah/my_project/Ibrahim_Dino_Unet/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        pretrained=False,
        freeze_backbone=False,
        decoder_channels=(512, 256, 128, 64),
        dropout=0.1,
        verbose=True,
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        y = model(x)

    print("[RESULT]")
    print("output shape:", y.shape)  # expected: (2, 9, 224, 224)