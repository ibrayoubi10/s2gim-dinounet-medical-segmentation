# author: Ibrahim (refactor) - Attention U-Net / U-Net
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.up(x)

class U_Net(nn.Module):
    def __init__(self, img_ch=3, num_class=9):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Conv1 = conv_block(img_ch, 64)
        self.Conv2 = conv_block(64, 128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)
        self.Conv5 = conv_block(512, 1024)
        self.Up5 = up_conv(1024, 512)
        self.Up_conv5 = conv_block(1024, 512)
        self.Up4 = up_conv(512, 256)
        self.Up_conv4 = conv_block(512, 256)
        self.Up3 = up_conv(256, 128)
        self.Up_conv3 = conv_block(256, 128)
        self.Up2 = up_conv(128, 64)
        self.Up_conv2 = conv_block(128, 64)
        self.Conv_1x1 = nn.Conv2d(64, num_class, 1)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1); x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2); x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3); x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4); x5 = self.Conv5(x5)
        d5 = self.Up5(x5); d5 = torch.cat((x4, d5), dim=1); d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5); d4 = torch.cat((x3, d4), dim=1); d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4); d3 = torch.cat((x2, d3), dim=1); d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((x1, d2), dim=1); d2 = self.Up_conv2(d2)
        logits = self.Conv_1x1(d2)  # (N, C, H, W)
        return logits