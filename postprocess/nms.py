import torch


def box_iou(a, b):
    """
    Vectorised IoU between boxes a (N, 4) and b (M, 4), xyxy format.
    Returns (N, M) IoU matrix.
    """
    a = a[:, None]   # (N, 1, 4)
    b = b[None]      # (1, M, 4)

    inter = (
        (torch.min(a[..., 2], b[..., 2]) - torch.max(a[..., 0], b[..., 0])).clamp(0) *
        (torch.min(a[..., 3], b[..., 3]) - torch.max(a[..., 1], b[..., 1])).clamp(0)
    )
    area_a = (a[..., 2] - a[..., 0]).clamp(0) * (a[..., 3] - a[..., 1]).clamp(0)
    area_b = (b[..., 2] - b[..., 0]).clamp(0) * (b[..., 3] - b[..., 1]).clamp(0)
    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes, scores, iou_thr=0.45, score_thr=0.25):
    """
    Pure-PyTorch greedy NMS.

    Args:
        boxes     : (N, 4) xyxy float tensor
        scores    : (N,)   float tensor
        iou_thr   : suppression threshold
        score_thr : minimum confidence to consider

    Returns:
        LongTensor of kept indices (into the filtered set).
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    mask   = scores > score_thr
    boxes  = boxes[mask]
    scores = scores[mask]

    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    order = scores.argsort(descending=True)
    kept  = []

    while order.numel() > 0:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break
        iou   = box_iou(boxes[order[:1]], boxes[order[1:]])[0]
        order = order[1:][iou < iou_thr]

    return torch.tensor(kept, dtype=torch.long)
