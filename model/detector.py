import torch
import torch.nn as nn
from model.fpn_neck import FPNNeck, ConvBnRelu


class DetectionHead(nn.Module):
    """Single-scale detection head: objectness + bbox regression."""

    def __init__(self, in_ch, num_anchors=3):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, in_ch, k=3),
            ConvBnRelu(in_ch, in_ch, k=3),
        )
        self.obj  = nn.Conv2d(in_ch, num_anchors,     1)
        self.bbox = nn.Conv2d(in_ch, num_anchors * 4, 1)

    def forward(self, x):
        feat = self.conv(x)
        return self.obj(feat), self.bbox(feat)


class PedestrianDetector(nn.Module):
    """
    MobileNetV2 + FPN backbone with three detection heads at P3/P4/P5.
    Returns dict with 'objectness' and 'boxes' tensors (concatenated across scales).
    """

    # Redesigned anchors targeting small objects (P3), medium (P4), large (P5)
    ANCHORS = {
        "p3": [(16, 32), (24, 56), (32, 80)],
        "p4": [(48, 96), (64, 128), (96, 160)],
        "p5": [(128, 256), (192, 384), (256, 512)],
    }

    def __init__(self, num_anchors=3, fpn_ch=256, pretrained=True):
        super().__init__()
        self.backbone = FPNNeck(out_ch=fpn_ch, pretrained=pretrained)
        self.head_p3  = DetectionHead(fpn_ch, num_anchors)
        self.head_p4  = DetectionHead(fpn_ch, num_anchors)
        self.head_p5  = DetectionHead(fpn_ch, num_anchors)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        obj3, box3 = self.head_p3(p3)
        obj4, box4 = self.head_p4(p4)
        obj5, box5 = self.head_p5(p5)

        def _flat(t):
            B = t.shape[0]
            return t.permute(0, 2, 3, 1).reshape(B, -1, t.shape[1] // t.shape[1])

        def _flat_obj(t):
            B = t.shape[0]
            return t.permute(0, 2, 3, 1).reshape(B, -1)

        def _flat_box(t, na=3):
            B = t.shape[0]
            return t.permute(0, 2, 3, 1).reshape(B, -1, 4)

        objectness = torch.cat([
            _flat_obj(obj3), _flat_obj(obj4), _flat_obj(obj5)
        ], dim=1)

        boxes = torch.cat([
            _flat_box(box3), _flat_box(box4), _flat_box(box5)
        ], dim=1)

        return {"objectness": objectness, "boxes": boxes}
