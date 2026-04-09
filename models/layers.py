"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        assert 0.0 <= p < 1.0, f"Dropout probability must be in [0, 1), received {p}"
        self.drop_rate = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        if not self.training or self.drop_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_rate
        binary_mask = (torch.rand(x.shape, device=x.device, dtype=x.dtype) < keep_prob).to(x.dtype)
        return x * binary_mask / keep_prob