"""
PINN Network for Mesh Generation.

Network architecture: (xi, eta, zeta) -> (x, y, z)
With input enhancement and tanh activation.
"""

import torch
import torch.nn as nn

# Add src to path for imports
import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from input_enhancement import InputEnhancement


class PINNNetwork(nn.Module):
    """PINN Network for (xi, eta, zeta) -> (x, y, z) mapping."""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=4,
        activation='tanh'
    ):
        """
        Initialize PINN Network.

        Args:
            input_dim: Input dimension (default 3 for xi, eta, zeta)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default 3 for x, y, z)
            num_layers: Number of hidden layers
            activation: Activation function ('tanh' or 'relu')
        """
        super().__init__()

        self.use_enhancement = False  # 暂时禁用
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        # Build layers - 原始3维输入
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

        # Output layer: 3D coordinates (x, y, z)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 初始化输出层
        nn.init.zeros_(self.output_layer.bias)
        # 缩放权重到合理范围
        with torch.no_grad():
            self.output_layer.weight *= 0.01

    def forward(self, xi, eta, zeta):
        """
        Forward pass.

        Args:
            xi: Input tensor (batch_size,)
            eta: Input tensor (batch_size,)
            zeta: Input tensor (batch_size,)

        Returns:
            Output tensor (batch_size, 3) representing (x, y, z)
        """
        # 原始输入
        x = torch.stack([xi, eta, zeta], dim=1)  # (batch, 3)

        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)

        # Output: physical coordinates (x, y, z)
        output = self.output_layer(x)  # (batch, 3)

        return output.T  # (3, batch) for compatibility

    def __repr__(self):
        return f"PINNNetwork(hidden_dim={self.hidden_dim}, num_layers={self.num_layers})"