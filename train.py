import os
import sys
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.detector import PedestrianDetector
from loss.ciou_loss  import ciou_loss


def train(cfg):
    """
    Minimal training loop.

    cfg attributes expected:
        data_root : str   — path to COCO-style dataset
        bs        : int   — batch size
        epochs    : int   — total epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # -- dataset (requires COCOPersonDataset in data/coco_person.py) --
    from data.coco_person import COCOPersonDataset
    ds = COCOPersonDataset(
        cfg.data_root,
        split="train2017",
        mosaic=True,
        img_size=640,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=ds.collate_fn,
    )

    model  = PedestrianDetector().to(device)
    opt    = SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        nesterov=True,
    )
    sched  = CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0

        for imgs, targets in loader:
            imgs    = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds    = model(imgs)
                tgt_box  = torch.cat([t["boxes"]      for t in targets], dim=0)
                tgt_obj  = torch.cat([t["objectness"] for t in targets], dim=0)

                loss_box = ciou_loss(
                    preds["boxes"].view(-1, 4),
                    tgt_box.view(-1, 4),
                )
                loss_obj = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds["objectness"].view(-1),
                    tgt_obj.view(-1).float(),
                )
                loss = loss_box + loss_obj

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        sched.step()
        avg = running / max(len(loader), 1)
        print(f"[train] epoch {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")

    return model
