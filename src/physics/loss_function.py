"""
PINN Loss Function.

Total loss: L = L_equation + L_boundary
- L_equation: Navier-Lamé equation constraint
- L_boundary: Boundary condition constraint
"""

import torch
import sys
import os

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from navier_lame_3d import NavierLame3D


class PINNLoss:
    """PINN Loss: L = L_equation + L_boundary"""

    def __init__(self, lambda_=1.0, mu=0.35, eq_weight=1.0, bc_weight=1.0):
        """
        Initialize PINN loss.

        Args:
            lambda_: Lamé constant (default 1.0)
            mu: Lamé constant (default 0.35)
            eq_weight: Weight for equation loss (default 1.0)
            bc_weight: Weight for boundary loss (default 1.0)
        """
        self.lambda_ = lambda_
        self.mu = mu
        self.eq_weight = eq_weight
        self.bc_weight = bc_weight
        self.physics = NavierLame3D(lambda_, mu)

    def compute_equation_loss(self, x, y, z, xi, eta, zeta):
        """
        Compute equation constraint loss.

        L_equation = mean(||f_x||² + ||f_y||² + ||f_z||²)

        Args:
            x, y, z: Network output (batch,)
            xi, eta, zeta: Input coordinates (batch,)

        Returns:
            Equation loss (scalar)
        """
        residuals = self.physics.compute_residuals(x, y, z, xi, eta, zeta)

        # Mean squared residuals
        loss = torch.mean(residuals**2)
        return loss

    def compute_boundary_loss(self, x, y, z, x_bc, y_bc, z_bc):
        """
        Compute boundary constraint loss.

        L_boundary = mean(||x - x_bc||² + ||y - y_bc||² + ||z - z_bc||²)

        Args:
            x, y, z: Network output (batch,)
            x_bc, y_bc, z_bc: Boundary targets (batch,)

        Returns:
            Boundary loss (scalar)
        """
        loss_x = torch.mean((x - x_bc)**2)
        loss_y = torch.mean((y - y_bc)**2)
        loss_z = torch.mean((z - z_bc)**2)

        loss = loss_x + loss_y + loss_z
        return loss

    def compute_components(self, x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc):
        """
        Compute individual loss components.

        Returns:
            eq_loss, bc_loss: Tuple of (equation_loss, boundary_loss)
        """
        eq_loss = self.compute_equation_loss(x, y, z, xi, eta, zeta)
        bc_loss = self.compute_boundary_loss(x, y, z, x_bc, y_bc, z_bc)
        return eq_loss, bc_loss

    def __call__(self, x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc):
        """
        Compute total loss.

        Returns:
            Total loss (scalar)
        """
        eq_loss = self.compute_equation_loss(x, y, z, xi, eta, zeta)
        bc_loss = self.compute_boundary_loss(x, y, z, x_bc, y_bc, z_bc)

        total = self.eq_weight * eq_loss + self.bc_weight * bc_loss
        return total

    def __repr__(self):
        return f"PINNLoss(lambda_={self.lambda_}, mu={self.mu}, eq_w={self.eq_weight}, bc_w={self.bc_weight})"