import torch
import math


def ciou_loss(pred, target, eps=1e-7):
    """
    Complete IoU loss (Zheng et al., AAAI 2020).
    pred / target : (N, 4) in cx-cy-w-h format, normalised [0, 1].
    Returns scalar mean loss.
    """
    if pred.numel() == 0 or target.numel() == 0:
        return pred.sum() * 0.0

    # convert to x1y1x2y2
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    inter = (
        (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) *
        (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    )
    area_p = (p_x2 - p_x1).clamp(0) * (p_y2 - p_y1).clamp(0)
    area_t = (t_x2 - t_x1).clamp(0) * (t_y2 - t_y1).clamp(0)
    union  = area_p + area_t - inter + eps
    iou    = inter / union

    # enclosing box diagonal squared
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    c2   = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # centre-point distance squared
    rho2 = (
        (pred[:, 0] - target[:, 0]) ** 2 +
        (pred[:, 1] - target[:, 1]) ** 2
    )

    # aspect-ratio consistency term v
    v = (4 / (math.pi ** 2)) * (
        torch.atan(target[:, 2] / (target[:, 3] + eps)) -
        torch.atan(pred[:, 2]  / (pred[:, 3]  + eps))
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()
