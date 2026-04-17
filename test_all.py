"""
test_all.py — full test suite for the pedestrian detector codebase.
Run from the project root:  python test_all.py
"""
import sys
import os
import math
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CIoU loss tests
# ---------------------------------------------------------------------------
class TestCIoULoss(unittest.TestCase):

    def setUp(self):
        from loss.ciou_loss import ciou_loss
        self.ciou_loss = ciou_loss

    def test_perfect_match_is_zero(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        loss  = self.ciou_loss(boxes, boxes.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=4,
                               msg="Identical pred/target should give ~0 loss")

    def test_no_overlap_is_near_one(self):
        pred   = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        loss   = self.ciou_loss(pred, target)
        self.assertGreater(loss.item(), 0.5,
                           msg="Non-overlapping boxes should yield high loss")

    def test_output_is_scalar(self):
        pred   = torch.rand(8, 4).abs()
        target = torch.rand(8, 4).abs()
        loss   = self.ciou_loss(pred, target)
        self.assertEqual(loss.shape, torch.Size([]),
                         msg="Loss must be a scalar tensor")

    def test_gradient_flows(self):
        pred   = torch.rand(4, 4, requires_grad=True)
        target = torch.rand(4, 4)
        loss   = self.ciou_loss(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad, msg="Gradients must reach pred tensor")
        self.assertFalse(pred.grad.isnan().any(), msg="Gradients must not be NaN")

    def test_empty_tensor_returns_zero(self):
        pred   = torch.zeros(0, 4, requires_grad=True)
        target = torch.zeros(0, 4)
        loss   = self.ciou_loss(pred, target)
        self.assertEqual(loss.item(), 0.0)

    def test_batch_of_sixteen(self):
        pred   = torch.rand(16, 4)
        target = torch.rand(16, 4)
        loss   = self.ciou_loss(pred, target)
        self.assertTrue(torch.isfinite(loss), msg="Loss should be finite for batch of 16")


# ---------------------------------------------------------------------------
# NMS tests
# ---------------------------------------------------------------------------
class TestNMS(unittest.TestCase):

    def setUp(self):
        from postprocess.nms import nms, box_iou
        self.nms     = nms
        self.box_iou = box_iou

    def test_keeps_best_of_two_overlapping(self):
        boxes  = torch.tensor([
            [10., 10., 50., 50.],
            [12., 12., 52., 52.],
        ])
        scores = torch.tensor([0.9, 0.8])
        kept   = self.nms(boxes, scores, iou_thr=0.45)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].item(), 0, msg="Highest-score box should be kept")

    def test_keeps_non_overlapping(self):
        boxes  = torch.tensor([
            [0.,  0.,  20., 20.],
            [100., 100., 120., 120.],
        ])
        scores = torch.tensor([0.9, 0.85])
        kept   = self.nms(boxes, scores, iou_thr=0.45)
        self.assertEqual(len(kept), 2, msg="Non-overlapping boxes should both be kept")

    def test_score_threshold_filters(self):
        boxes  = torch.tensor([[0., 0., 10., 10.]])
        scores = torch.tensor([0.1])
        kept   = self.nms(boxes, scores, score_thr=0.25)
        self.assertEqual(len(kept), 0, msg="Low-score box should be filtered out")

    def test_empty_input(self):
        boxes  = torch.zeros(0, 4)
        scores = torch.zeros(0)
        kept   = self.nms(boxes, scores)
        self.assertEqual(len(kept), 0)

    def test_box_iou_self(self):
        boxes = torch.tensor([[0., 0., 10., 10.]])
        iou   = self.box_iou(boxes, boxes)
        self.assertAlmostEqual(iou[0, 0].item(), 1.0, places=4)

    def test_box_iou_no_overlap(self):
        a = torch.tensor([[0., 0., 5., 5.]])
        b = torch.tensor([[10., 10., 15., 15.]])
        self.assertAlmostEqual(self.box_iou(a, b)[0, 0].item(), 0.0, places=4)

    def test_returns_long_tensor(self):
        boxes  = torch.rand(5, 4)
        scores = torch.rand(5)
        kept   = self.nms(boxes, scores)
        self.assertEqual(kept.dtype, torch.long)


# ---------------------------------------------------------------------------
# Mosaic augmentation tests
# ---------------------------------------------------------------------------
class TestMosaic(unittest.TestCase):

    def setUp(self):
        from data.mosaic_aug import mosaic4
        self.mosaic4 = mosaic4

    def _rand_img(self, h=200, w=200):
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def _rand_boxes(self, n=3, img_w=200, img_h=200):
        x1 = np.random.randint(0, img_w - 20, (n, 1)).astype(np.float32)
        y1 = np.random.randint(0, img_h - 20, (n, 1)).astype(np.float32)
        x2 = x1 + np.random.randint(10, 40, (n, 1)).astype(np.float32)
        y2 = y1 + np.random.randint(10, 40, (n, 1)).astype(np.float32)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        cls = np.zeros((n, 1), dtype=np.float32)
        return np.hstack([cls, x1, y1, x2, y2])

    def test_output_shape(self):
        imgs  = [self._rand_img() for _ in range(4)]
        boxes = [self._rand_boxes() for _ in range(4)]
        out, merged = self.mosaic4(imgs, boxes, img_size=640)
        self.assertEqual(out.shape, (640, 640, 3))

    def test_merged_boxes_within_bounds(self):
        imgs  = [self._rand_img() for _ in range(4)]
        boxes = [self._rand_boxes() for _ in range(4)]
        out, merged = self.mosaic4(imgs, boxes, img_size=320)
        if len(merged) > 0:
            self.assertTrue((merged[:, 1:] >= 0).all())
            self.assertTrue((merged[:, 1:] <= 320).all())

    def test_empty_boxes_list(self):
        imgs  = [self._rand_img() for _ in range(4)]
        boxes = [np.zeros((0, 5), dtype=np.float32) for _ in range(4)]
        out, merged = self.mosaic4(imgs, boxes)
        self.assertEqual(merged.shape[0], 0)

    def test_output_dtype(self):
        imgs  = [self._rand_img() for _ in range(4)]
        boxes = [self._rand_boxes() for _ in range(4)]
        out, _ = self.mosaic4(imgs, boxes)
        self.assertEqual(out.dtype, np.uint8)


# ---------------------------------------------------------------------------
# FPN neck tests
# ---------------------------------------------------------------------------
class TestFPNNeck(unittest.TestCase):

    def setUp(self):
        from model.fpn_neck import FPNNeck
        self.FPNNeck = FPNNeck

    def test_output_count(self):
        neck = self.FPNNeck(out_ch=128, pretrained=False)
        x    = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            outs = neck(x)
        self.assertEqual(len(outs), 3, msg="FPN should return 3 feature maps")

    def test_channel_count(self):
        out_ch = 64
        neck   = self.FPNNeck(out_ch=out_ch, pretrained=False)
        x      = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            p3, p4, p5 = neck(x)
        for fi, fm in enumerate([p3, p4, p5]):
            self.assertEqual(fm.shape[1], out_ch,
                             msg=f"P{fi+3} should have {out_ch} channels")

    def test_stride_ordering(self):
        neck = self.FPNNeck(out_ch=64, pretrained=False)
        x    = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            p3, p4, p5 = neck(x)
        self.assertGreater(p3.shape[-1], p4.shape[-1])
        self.assertGreater(p4.shape[-1], p5.shape[-1])

    def test_no_nan_in_forward(self):
        neck = self.FPNNeck(out_ch=64, pretrained=False)
        x    = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            for fm in neck(x):
                self.assertFalse(fm.isnan().any(), msg="NaN detected in FPN output")


# ---------------------------------------------------------------------------
# Full detector tests
# ---------------------------------------------------------------------------
class TestDetector(unittest.TestCase):

    def setUp(self):
        from model.detector import PedestrianDetector
        self.model = PedestrianDetector(pretrained=False)
        self.model.eval()

    def test_forward_returns_dict(self):
        x = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            out = self.model(x)
        self.assertIn("objectness", out)
        self.assertIn("boxes", out)

    def test_objectness_shape(self):
        x = torch.zeros(2, 3, 320, 320)
        with torch.no_grad():
            out = self.model(x)
        B = out["objectness"].shape[0]
        self.assertEqual(B, 2, msg="Batch dim must match input")

    def test_boxes_last_dim(self):
        x = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out["boxes"].shape[-1], 4,
                         msg="Each predicted box must have 4 coordinates")

    def test_no_nan_in_output(self):
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = self.model(x)
        for k, v in out.items():
            self.assertFalse(v.isnan().any(), msg=f"NaN in detector output['{k}']")

    def test_different_input_sizes(self):
        for sz in [320, 416, 640]:
            x = torch.zeros(1, 3, sz, sz)
            with torch.no_grad():
                out = self.model(x)
            self.assertIn("boxes", out, msg=f"Forward failed for img_size={sz}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suites = [
        loader.loadTestsFromTestCase(TestCIoULoss),
        loader.loadTestsFromTestCase(TestNMS),
        loader.loadTestsFromTestCase(TestMosaic),
        loader.loadTestsFromTestCase(TestFPNNeck),
        loader.loadTestsFromTestCase(TestDetector),
    ]
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestSuite(suites))
    sys.exit(0 if result.wasSuccessful() else 1)
