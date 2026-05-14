import pytest
import torch
import sys
sys.path.insert(0, 'src/network')
from pinn_network import PINNNetwork


def test_pinn_init():
    """Test PINN initialization"""
    model = PINNNetwork(input_dim=3, hidden_dim=32, output_dim=3, num_layers=2)
    assert model is not None


def test_forward_shape():
    """Test forward pass produces correct output shape"""
    model = PINNNetwork(input_dim=3, hidden_dim=64, output_dim=3, num_layers=4)
    batch_size = 10
    xi = torch.randn(batch_size)
    eta = torch.randn(batch_size)
    zeta = torch.randn(batch_size)

    output = model(xi, eta, zeta)

    assert output.shape[0] == 3
    assert output.shape[1] == batch_size


def test_output_is_coordinates():
    """Test output is physical coordinates"""
    model = PINNNetwork(input_dim=3, hidden_dim=32, output_dim=3, num_layers=2)
    # Identity mapping test
    xi = torch.tensor([0.0, 0.5, 1.0])
    eta = torch.tensor([0.0, 0.0, 0.0])
    zeta = torch.tensor([0.0, 0.0, 0.0])

    output = model(xi, eta, zeta)

    # Output should be 3D coordinates
    assert output.shape[0] == 3


def test_requires_grad():
    """Test that output requires grad for backprop"""
    model = PINNNetwork(input_dim=3, hidden_dim=32, output_dim=3, num_layers=2)
    xi = torch.randn(10, requires_grad=True)
    eta = torch.randn(10, requires_grad=True)
    zeta = torch.randn(10, requires_grad=True)

    output = model(xi, eta, zeta)

    # Check that output can require grad
    assert output.requires_grad