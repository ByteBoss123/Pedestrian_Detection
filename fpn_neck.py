import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPNNeck(nn.Module):
    """Feature Pyramid Network over MobileNetV2 C3/C4/C5."""

    def __init__(self, out_ch=256, pretrained=True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights).features
        # C3 -> stride 8,  C4 -> stride 16,  C5 -> stride 32
        self.c3 = backbone[:7]
        self.c4 = backbone[7:14]
        self.c5 = backbone[14:]

        self.lat5 = ConvBnRelu(1280, out_ch)
        self.lat4 = ConvBnRelu(96,   out_ch)
        self.lat3 = ConvBnRelu(32,   out_ch)

        self.out5 = ConvBnRelu(out_ch, out_ch, k=3)
        self.out4 = ConvBnRelu(out_ch, out_ch, k=3)
        self.out3 = ConvBnRelu(out_ch, out_ch, k=3)

    def forward(self, x):
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        return self.out3(p3), self.out4(p4), self.out5(p5)  # P3 P4 P5
