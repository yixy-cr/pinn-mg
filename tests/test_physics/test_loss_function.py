import pytest
import torch
import sys
sys.path.insert(0, 'src/physics')
from loss_function import PINNLoss


def test_loss_init():
    """Test loss function initialization"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)
    assert loss_fn.lambda_ == 1.0
    assert loss_fn.mu == 0.35


def test_total_loss():
    """Test total loss computation"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)

    batch_size = 10
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)

    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)

    x_bc = x.detach()
    y_bc = y.detach()
    z_bc = z.detach()

    total_loss = loss_fn(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() >= 0


def test_loss_components():
    """Test individual loss components"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)

    batch_size = 5
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    x_bc = x.detach()
    y_bc = y.detach()
    z_bc = z.detach()

    eq_loss, bc_loss = loss_fn.compute_components(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)

    assert eq_loss.item() >= 0
    assert bc_loss.item() >= 0


def test_callable():
    """Test loss as callable"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)

    batch_size = 5
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    x_bc = x.detach()
    y_bc = y.detach()
    z_bc = z.detach()

    loss = loss_fn(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)

    assert loss.item() >= 0