"""Unified multi-task model (root-level for autograder import)

The autograder does: from multitask import MultiTaskPerceptionModel
"""

import os
import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


IMG_DIM = 224


def _safe_load(path, device="cpu"):
    """Load checkpoint, handling both raw state_dict and wrapper formats."""
    _ver = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    if _ver >= (1, 13):
        data = torch.load(path, map_location=device, weights_only=True)
    else:
        data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
        import gdown
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)

        super().__init__()
        self.img_sz = IMG_DIM

        # Instantiate the three task models
        self.cls_net = VGG11Classifier(num_classes=num_breeds, dropout_p=0.5)
        self.loc_net = VGG11Localizer(in_channels=in_channels, dropout_p=0.5, freeze_backbone=False)
        self.seg_net = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load trained weights into each
        self._restore_weights(self.cls_net, classifier_path, tag="classifier")
        self._restore_weights(self.loc_net, localizer_path, tag="localizer")
        self._restore_weights(self.seg_net, unet_path, tag="segmentation")

    def _restore_weights(self, model, ckpt_path, tag=""):
        if not os.path.exists(ckpt_path):
            print(f"  [{tag}] checkpoint missing: {ckpt_path} — using random init")
            return
        weights = _safe_load(ckpt_path)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        clean = {k.replace("module.", ""): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print(f"  [{tag}] loaded | missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor in pixel space.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        cls_out = self.cls_net(x)
        loc_out = self.loc_net(x) * float(self.img_sz)
        seg_out = self.seg_net(x) + 1

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }