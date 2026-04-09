"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

POOL_SIZE = 4


class BBoxHead(nn.Module):
    """Compact MLP head that predicts normalized bounding box coordinates."""

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((POOL_SIZE, POOL_SIZE))
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * POOL_SIZE * POOL_SIZE, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for hidden layers, small gain for output layer
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.constant_(self.layers[1].bias, 0.0)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.constant_(self.layers[4].bias, 0.0)
        nn.init.xavier_uniform_(self.layers[7].weight, gain=0.1)
        nn.init.constant_(self.layers[7].bias, 0.0)

    def forward(self, feat_map):
        pooled = self.avg_pool(feat_map)
        raw = self.layers(pooled)
        return torch.sigmoid(raw)


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout_p: float = 0.5, freeze_backbone: bool = True):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.bbox_head = BBoxHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format
            normalized to [0, 1].
        """
        _, skip_feats = self.encoder(x, return_features=True)
        return self.bbox_head(skip_feats["f5"])