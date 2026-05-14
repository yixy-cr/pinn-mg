import pytest
import torch
import sys
sys.path.insert(0, 'src/network')
from input_enhancement import InputEnhancement


def test_enhance_init():
    """Test enhancement initialization"""
    enhancer = InputEnhancement()
    assert enhancer is not None


def test_enhance_3d():
    """Test 3D input enhancement"""
    enhancer = InputEnhancement()
    batch_size = 10
    xi = torch.linspace(0, 1, batch_size)
    eta = torch.linspace(0, 1, batch_size)
    zeta = torch.linspace(0, 1, batch_size)

    result = enhancer.enhance(xi, eta, zeta)

    # Should be: [xi, eta, zeta, tan(xi), tan(eta), tan(zeta), cot(xi), cot(eta), cot(zeta)]
    assert result.shape[0] == 9
    assert result.shape[1] == batch_size


def test_tan_values():
    """Test tan calculations"""
    enhancer = InputEnhancement()
    xi = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    eta = torch.zeros(5)
    zeta = torch.zeros(5)

    result = enhancer.enhance(xi, eta, zeta)

    # tan(0) = 0, tan(0.5) ≈ 0.546 (not inf because we handle edge cases)
    assert result.shape[0] == 9
    # Check all values are finite
    assert torch.isfinite(result).all()


def test_cot_values():
    """Test cot calculations with edge case handling"""
    enhancer = InputEnhancement()
    # Test at boundaries where tan approaches 0
    xi = torch.tensor([0.001, 0.5, 1.0])
    eta = torch.zeros(3)
    zeta = torch.zeros(3)

    result = enhancer.enhance(xi, eta, zeta)

    # cot(xi) = 1/tan(xi), should handle near-zero
    assert torch.isfinite(result).all()


def test_forward():
    """Test forward method"""
    enhancer = InputEnhancement()
    xi = torch.tensor([0.0, 0.5, 1.0])
    eta = torch.tensor([0.0, 0.5, 1.0])
    zeta = torch.tensor([0.0, 0.5, 1.0])

    result = enhancer(xi, eta, zeta)

    assert result.shape[0] == 9