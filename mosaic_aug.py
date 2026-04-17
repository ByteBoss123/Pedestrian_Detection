import random
import numpy as np
import cv2


def mosaic4(imgs, boxes_list, img_size=640):
    """
    Stitch 4 images into one mosaic tile.

    Args:
        imgs       : list of 4 BGR ndarrays (any size).
        boxes_list : list of 4 arrays of shape (N, 5) — [cls, x1, y1, x2, y2]
                     in pixel coords relative to each source image.
        img_size   : final output square size (pixels).

    Returns:
        mosaic  : ndarray of shape (img_size, img_size, 3)
        merged  : ndarray of shape (M, 5) with boxes in output pixel coords
    """
    s  = img_size
    cx = random.randint(s // 4, 3 * s // 4)
    cy = random.randint(s // 4, 3 * s // 4)

    canvas       = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
    merged_boxes = []

    placements = [
        (0,  0,  cx,    cy),     # top-left
        (cx, 0,  2 * s, cy),     # top-right
        (0,  cy, cx,    2 * s),  # bottom-left
        (cx, cy, 2 * s, 2 * s),  # bottom-right
    ]

    for i, (x1, y1, x2, y2) in enumerate(placements):
        tile_w = x2 - x1
        tile_h = y2 - y1
        img    = cv2.resize(imgs[i], (tile_w, tile_h))
        canvas[y1:y2, x1:x2] = img

        bx = boxes_list[i]
        if bx is None or len(bx) == 0:
            continue
        bx     = bx.copy().astype(np.float32)
        oh, ow = imgs[i].shape[:2]
        sx     = tile_w / ow
        sy     = tile_h / oh
        bx[:, 1] = bx[:, 1] * sx + x1
        bx[:, 2] = bx[:, 2] * sy + y1
        bx[:, 3] = bx[:, 3] * sx + x1
        bx[:, 4] = bx[:, 4] * sy + y1
        merged_boxes.append(bx)

    mosaic = cv2.resize(canvas, (s, s))
    scale  = s / (2 * s)

    if merged_boxes:
        merged = np.concatenate(merged_boxes, axis=0)
        merged[:, 1:] *= scale
        # clip to image bounds
        merged[:, 1] = np.clip(merged[:, 1], 0, s)
        merged[:, 2] = np.clip(merged[:, 2], 0, s)
        merged[:, 3] = np.clip(merged[:, 3], 0, s)
        merged[:, 4] = np.clip(merged[:, 4], 0, s)
        # drop degenerate boxes
        keep   = (merged[:, 3] - merged[:, 1] > 1) & (merged[:, 4] - merged[:, 2] > 1)
        merged = merged[keep]
    else:
        merged = np.zeros((0, 5), dtype=np.float32)

    return mosaic, merged
