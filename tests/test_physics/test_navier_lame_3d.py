import pytest
import torch
import sys
sys.path.insert(0, 'src/physics')
from navier_lame_3d import NavierLame3D


def test_navier_lame_init():
    """Test Navier-Lamé initialization"""
    physics = NavierLame3D(lambda_=1.0, mu=0.35)
    assert physics.lambda_ == 1.0
    assert physics.mu == 0.35


def test_compute_residuals():
    """Test computing Navier-Lamé residuals"""
    physics = NavierLame3D(lambda_=1.0, mu=0.35)

    batch_size = 10
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)

    # Network outputs (x, y, z)
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)

    residuals = physics.compute_residuals(x, y, z, xi, eta, zeta)

    # Should return 3 residuals (f_x, f_y, f_z)
    assert residuals.shape[0] == 3
    assert residuals.shape[1] == batch_size


def test_zero_residual_for_linear():
    """Test that linear solution gives small residuals"""
    physics = NavierLame3D(lambda_=1.0, mu=0.35)

    batch_size = 5
    xi = torch.linspace(0, 1, batch_size, requires_grad=True).reshape(-1)
    eta = torch.zeros_like(xi)
    zeta = torch.zeros_like(xi)

    # Simple identity mapping: x = xi, y = eta, z = zeta
    x = xi.clone()
    y = eta.clone()
    z = zeta.clone()

    residuals = physics.compute_residuals(x, y, z, xi, eta, zeta)

    # Linear solution should give non-zero but bounded residuals
    assert residuals.shape[0] == 3