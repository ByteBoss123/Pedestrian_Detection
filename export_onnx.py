import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.detector import PedestrianDetector


def export(ckpt_path=None, out_path="pedestrian.onnx", img_size=640):
    """
    Export PedestrianDetector to ONNX (opset 17) and probe CPU latency.

    Args:
        ckpt_path : path to .pt checkpoint, or None to use random weights.
        out_path  : destination .onnx file.
        img_size  : square input size (px).
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[export] onnx / onnxruntime not installed — skipping ONNX check.")
        onnx = None

    model = PedestrianDetector()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
    model.eval()

    dummy   = torch.zeros(1, 3, img_size, img_size)
    dynamic = {
        "images":      {0: "batch"},
        "objectness":  {0: "batch"},
        "boxes":       {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["images"],
        output_names=["objectness", "boxes"],
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export] saved -> {out_path}")

    if onnx is not None:
        onnx.checker.check_model(onnx.load(out_path))
        print("[export] onnx graph check passed")

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        inp  = dummy.numpy()
        # warm-up
        for _ in range(5):
            sess.run(None, {"images": inp})
        # timed runs
        N  = 50
        t0 = time.perf_counter()
        for _ in range(N):
            sess.run(None, {"images": inp})
        avg_ms = (time.perf_counter() - t0) / N * 1000
        status = "PASS" if avg_ms <= 50 else "FAIL"
        print(f"[export] avg latency: {avg_ms:.1f} ms  (target <=50ms)  [{status}]")

    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default=None)
    p.add_argument("--out",      default="pedestrian.onnx")
    p.add_argument("--img-size", default=640, type=int)
    args = p.parse_args()
    export(args.ckpt, args.out, args.img_size)
