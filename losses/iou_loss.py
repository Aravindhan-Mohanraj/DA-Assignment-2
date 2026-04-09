"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Invalid reduction mode: {reduction}"
        self.eps = eps
        self.reduction = reduction

    def _cxcywh_to_corners(self, boxes):
        """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
        half_w = boxes[:, 2] * 0.5
        half_h = boxes[:, 3] * 0.5
        x1 = boxes[:, 0] - half_w
        y1 = boxes[:, 1] - half_h
        x2 = boxes[:, 0] + half_w
        y2 = boxes[:, 1] + half_h
        return x1, y1, x2, y2

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        px1, py1, px2, py2 = self._cxcywh_to_corners(pred_boxes)
        gx1, gy1, gx2, gy2 = self._cxcywh_to_corners(target_boxes)

        # intersection
        ix1 = torch.max(px1, gx1)
        iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2)
        iy2 = torch.min(py2, gy2)
        intersection = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

        # union
        area_pred = (px2 - px1) * (py2 - py1)
        area_gt = (gx2 - gx1) * (gy2 - gy1)
        union = area_pred + area_gt - intersection + self.eps

        iou_scores = intersection / union
        loss_per_sample = 1.0 - iou_scores

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample