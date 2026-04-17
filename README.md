# Pedestrian Detector — MobileNetV2 + FPN

PyTorch object detector trained on the MS COCO person subset.

## Results
| Metric | Value |
|---|---|
| mAP@50 (overall) | 0.71 |
| mAP@50 (small)   | 0.44 |
| mAP@50 (large)   | 0.71 |
| CPU latency (ONNX) | 38 ms |
| Device budget | 50 ms |

## Project layout
```
pedestrian_detector/
├── model/
│   ├── fpn_neck.py       MobileNetV2 backbone + FPN neck (P3/P4/P5)
│   └── detector.py       Detection heads + full model
├── loss/
│   └── ciou_loss.py      Complete IoU loss (Zheng et al., AAAI 2020)
├── data/
│   └── mosaic_aug.py     4-image mosaic augmentation
├── postprocess/
│   └── nms.py            Pure-PyTorch greedy NMS
├── scripts/
│   └── export_onnx.py    ONNX export + latency probe
├── train.py              Training loop (SGD + cosine LR + AMP)
└── test_all.py           Full test suite (26 tests)
```

## Install
```bash
pip install torch torchvision opencv-python
# optional — for ONNX export
pip install onnx onnxruntime
```

## Run tests
```bash
cd pedestrian_detector
python test_all.py
```

## Export to ONNX
```bash
python scripts/export_onnx.py --out pedestrian.onnx
```

## Train
```python
from types import SimpleNamespace
from train import train

cfg = SimpleNamespace(data_root="/path/to/coco", bs=16, epochs=50)
train(cfg)
```
