import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/geometry')
from boundary_fitter import BoundaryFitter


def test_fitter_init():
    """Test fitter initialization"""
    fitter = BoundaryFitter()
    assert fitter.C == 1.0
    assert fitter.epsilon == 0.01


def test_fit_unit_cube():
    """Test SVR on unit cube boundary"""
    fitter = BoundaryFitter(C=10.0, epsilon=0.001)

    # Sample unit cube corners
    xi = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    eta = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    zeta = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    fitter.fit(xi, eta, zeta, x, y, z)

    assert fitter.model_x is not None
    assert fitter.model_y is not None
    assert fitter.model_z is not None


def test_predict():
    """Test prediction"""
    fitter = BoundaryFitter(C=10.0, epsilon=0.001)

    # Simple identity mapping
    n = 20
    xi = np.linspace(0, 1, n)
    eta = np.zeros(n)
    zeta = np.zeros(n)
    x = xi.copy()
    y = np.zeros(n)
    z = np.zeros(n)

    fitter.fit(xi, eta, zeta, x, y, z)

    # Test prediction at mid-point
    pred = fitter.predict(0.5, 0, 0)
    assert pred.shape == (3,)
    assert abs(pred[0] - 0.5) < 0.1


def test_callable():
    """Test callable interface"""
    fitter = BoundaryFitter(C=10.0, epsilon=0.001)

    n = 20
    xi = np.linspace(0, 1, n)
    eta = np.zeros(n)
    zeta = np.zeros(n)
    x = xi.copy()
    y = np.zeros(n)
    z = np.zeros(n)

    fitter.fit(xi, eta, zeta, x, y, z)

    # Use as callable
    result = fitter(0.5, 0, 0)
    assert result.shape == (3,)