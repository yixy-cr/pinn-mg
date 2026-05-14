"""
Input Enhancement for PINN Network.

Enhances input with tan/cot transformations:
U_in = [xi, eta, zeta, tan(xi), tan(eta), tan(zeta), cot(xi), cot(eta), cot(zeta)]
"""

import torch
import torch.nn as nn


class InputEnhancement(nn.Module):
    """Input enhancement with tan/cot transformation."""

    def __init__(self, epsilon=1e-10):
        """
        Initialize input enhancement.

        Args:
            epsilon: Small value to avoid division by zero in cot
        """
        super().__init__()
        self.epsilon = epsilon

    def safe_tan(self, x):
        """
        Safe tan function.

        Args:
            x: Input tensor

        Returns:
            tan(x)
        """
        return torch.tan(x)

    def safe_cot(self, x):
        """
        Safe cot function: cot(x) = 1/tan(x), handling edge cases.

        Args:
            x: Input tensor

        Returns:
            cot(x)
        """
        tan_x = torch.tan(x)
        # Avoid division by zero
        tan_x = torch.where(
            torch.abs(tan_x) < self.epsilon,
            torch.ones_like(tan_x) * self.epsilon,
            tan_x
        )
        return 1.0 / tan_x

    def enhance(self, xi, eta, zeta):
        """
        Enhance input with tan/cot transformations.

        Args:
            xi: Input tensor (batch_size,)
            eta: Input tensor (batch_size,)
            zeta: Input tensor (batch_size,)

        Returns:
            Enhanced input tensor (9, batch_size)
            [xi, eta, zeta, tan(xi), tan(eta), tan(zeta), cot(xi), cot(eta), cot(zeta)]
        """
        # Original coordinates
        coords = torch.stack([xi, eta, zeta], dim=0)

        # Tan transformations (逐维)
        tan_coords = torch.stack([
            self.safe_tan(xi),
            self.safe_tan(eta),
            self.safe_tan(zeta)
        ], dim=0)

        # Cot transformations (逐维)
        cot_coords = torch.stack([
            self.safe_cot(xi),
            self.safe_cot(eta),
            self.safe_cot(zeta)
        ], dim=0)

        # Concatenate: [xi, eta, zeta, tan, cot]
        enhanced = torch.cat([coords, tan_coords, cot_coords], dim=0)

        return enhanced

    def forward(self, xi, eta, zeta):
        """
        Forward pass.

        Args:
            xi, eta, zeta: Input tensors

        Returns:
            Enhanced input
        """
        return self.enhance(xi, eta, zeta)

    def __repr__(self):
        return f"InputEnhancement(epsilon={self.epsilon})"