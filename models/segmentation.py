"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11Encoder


class DecoderStage(nn.Module):
    """Single decoder stage: transposed conv upsample → concat skip → conv block."""

    def __init__(self, inp_ch, skip_ch, out_ch, double_conv=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inp_ch, out_ch, kernel_size=2, stride=2)

        merged_ch = out_ch + skip_ch
        conv_layers = [
            nn.Conv2d(merged_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if double_conv:
            conv_layers += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.refine = nn.Sequential(*conv_layers)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 1, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)

        # Bottleneck bridge at the bottom of the U
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Expanding path (mirrors encoder block structure)
        self.up5 = DecoderStage(512, 512, 512, double_conv=True)
        self.up4 = DecoderStage(512, 512, 256, double_conv=True)
        self.up3 = DecoderStage(256, 256, 128, double_conv=True)
        self.up2 = DecoderStage(128, 128, 64, double_conv=False)
        self.up1 = DecoderStage(64, 64, 64, double_conv=False)

        # Final upsample to restore original resolution + 1x1 classifier
        self.last_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        _, skips = self.encoder(x, return_features=True)

        bridge = self.bottleneck(skips["f5"])

        o5 = self.up5(bridge, skips["f5"])
        o4 = self.up4(o5, skips["f4"])
        o3 = self.up3(o4, skips["f3"])
        o2 = self.up2(o3, skips["f2"])
        o1 = self.up1(o2, skips["f1"])

        o1 = self.last_upsample(o1)
        return self.seg_head(o1)