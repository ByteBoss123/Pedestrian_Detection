# Pedestrian Detector Components: MobileNetV2 + FPN, CIoU Loss, Mosaic, NMS

A from-scratch implementation of the building blocks of a single-stage person detector in PyTorch, with a unit
test for each and an ONNX export path for CPU deployment.

## Status: components, not a trained detector

An earlier version of this README reported mAP@50 of 0.71 on COCO persons. **There is no evidence that a training
run happened**: `train.py` imports a COCO dataset loader (`data/coco_person.py`) that was never written, no
checkpoint or evaluation output exists, and the README offered no log. Those numbers have been removed. What is
here and verified:

| Component | File | Verified by |
|---|---|---|
| MobileNetV2 backbone + FPN neck (P3/P4/P5) | `model/fpn_neck.py` | output count, channels and stride order tests |
| Anchor-based detection heads | `model/detector.py` | output shapes, no NaNs, variable input sizes |
| Complete-IoU loss (Zheng et al., AAAI 2020) | `loss/ciou_loss.py` | identical boxes give 0, disjoint boxes give high loss, gradients flow |
| 4-image mosaic augmentation | `data/mosaic_aug.py` | output shape, box remapping and clipping |
| Greedy NMS in pure PyTorch | `postprocess/nms.py` | matches expected suppression, IoU correctness |
| ONNX export + CPU latency probe | `scripts/export_onnx.py` | ONNX graph check, onnxruntime inference |

```
python test_all.py       # 26 tests, all pass (October 2026)
python scripts/export_onnx.py --out pedestrian.onnx
```

## What is needed to make it a detector

1. `data/coco_person.py`: a COCO person-only dataset that feeds `mosaic4` and produces anchor targets.
2. A GPU training run (`train.py`) and an evaluation script reporting COCO-style mAP@50 by object size.
3. Re-run the ONNX latency probe on the trained checkpoint.

Until then, this repo shows detector engineering (architecture, loss, augmentation, post-processing, export), not
detection accuracy.
