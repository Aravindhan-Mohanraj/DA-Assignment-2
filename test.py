# test_multitask_shapes.py
import torch
from models.multitask import MultiTaskPerceptionModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskPerceptionModel(seg_classes=3).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        out = model(x)

    cls = out["classification"]
    loc = out["localization"]
    seg = out["segmentation"]  # raw logits (B,C,H,W)

    print(f"classification: {cls.shape}")
    print(f"localization:   {loc.shape}")
    print(f"segmentation:   {seg.shape}")

    # expected: (B, 3, 224, 224)
    assert seg.ndim == 4, "segmentation must be (B,C,H,W)"
    assert seg.shape[1] == 3, "segmentation must have 3 channels (trimap)"

    # show how to convert to trimap 1/2/3
    pred = seg.argmax(1)      # (B,H,W) in {0,1,2}
    pred_trimap = pred + 1    # (B,H,W) in {1,2,3}
    print(f"pred shape: {pred.shape} | trimap min/max: {pred_trimap.min().item()}/{pred_trimap.max().item()}")

if __name__ == "__main__":
    main()

