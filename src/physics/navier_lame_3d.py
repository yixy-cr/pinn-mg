"""
3D Navier-Lamé Equation in Computational Domain.

The equation: (λ+μ)∇(∇·u) + μ∇²u = 0
where u = (x-ξ, y-η, z-ζ) is displacement.

Computing in computational domain (ξ, η, ζ) using automatic differentiation.
"""

import torch


class NavierLame3D:
    """
    3D Navier-Lamé equation in computational domain.

    The equation is expressed in terms of derivatives with respect to
    computational coordinates (xi, eta, zeta).
    """

    def __init__(self, lambda_=1.0, mu=0.35):
        """
        Initialize Navier-Lamé physics.

        Args:
            lambda_: First Lamé constant (default 1.0)
            mu: Second Lamé constant (default 0.35)
        """
        self.lambda_ = lambda_
        self.mu = mu

    def compute_first_derivatives(self, f, x):
        """
        Compute first derivative df/dx using autograd.

        Args:
            f: Function values (batch,) - must require grad
            x: Independent variable (batch,) - must require grad

        Returns:
            df_dx: First derivative
        """
        # Ensure f requires grad
        if not f.requires_grad:
            f = f.requires_grad_(True)

        df_dx = torch.autograd.grad(
            f.sum(), x, create_graph=True, retain_graph=True, allow_unused=True
        )[0]

        # Handle case where gradient is None
        if df_dx is None:
            df_dx = torch.zeros_like(x)

        return df_dx

    def compute_second_derivatives(self, f, x):
        """
        Compute second derivative d²f/dx² using autograd.

        Args:
            f: First derivative values (batch,) - must require grad
            x: Independent variable (batch,) - must require grad

        Returns:
            d2f_dx2: Second derivative
        """
        # Ensure f requires grad
        if not f.requires_grad:
            f = f.requires_grad_(True)

        d2f_dx2 = torch.autograd.grad(
            f.sum(), x, create_graph=True, retain_graph=True, allow_unused=True
        )[0]

        # Handle case where gradient is None
        if d2f_dx2 is None:
            d2f_dx2 = torch.zeros_like(x)

        return d2f_dx2

    def compute_residuals(self, x, y, z, xi, eta, zeta):
        """
        Compute Navier-Lamé residuals.

        The equation in 3D computational domain:
        f_x = (λ+μ) * d(x-ξ)/dxi * d(d(x-ξ)/dxi + ...)/dxi + μ * d²(x-ξ)/dxi²
        etc.

        Note: x, y, z should be functions of xi, eta, zeta (output from PINN network)

        Args:
            x, y, z: Network output (batch,) - physical coordinates
            xi, eta, zeta: Input coordinates (batch,) - computational domain

        Returns:
            residuals: (3, batch) for f_x, f_y, f_z
        """
        # Define displacement u = (x - xi, y - eta, z - zeta)
        u_x = x - xi
        u_y = y - eta
        u_z = z - zeta

        # First derivatives of displacement
        du_x_dxi = self.compute_first_derivatives(u_x, xi)
        du_y_deta = self.compute_first_derivatives(u_y, eta)
        du_z_dzeta = self.compute_first_derivatives(u_z, zeta)

        # Divergence of displacement
        div_u = du_x_dxi + du_y_deta + du_z_dzeta

        # Second derivatives (Laplacian components)
        d2u_x_dxi2 = self.compute_second_derivatives(du_x_dxi, xi)
        d2u_y_deta2 = self.compute_second_derivatives(du_y_deta, eta)
        d2u_z_dzeta2 = self.compute_second_derivatives(du_z_dzeta, zeta)

        # Navier-Lamé equation: (λ+μ)∇(∇·u) + μ∇²u = 0
        f_x = (self.lambda_ + self.mu) * div_u * du_x_dxi + self.mu * d2u_x_dxi2
        f_y = (self.lambda_ + self.mu) * div_u * du_y_deta + self.mu * d2u_y_deta2
        f_z = (self.lambda_ + self.mu) * div_u * du_z_dzeta + self.mu * d2u_z_dzeta2

        residuals = torch.stack([f_x, f_y, f_z])

        return residuals

    def __repr__(self):
        return f"NavierLame3D(lambda_={self.lambda_}, mu={self.mu})"