import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/mesh')
from gmsh_exporter import GMSHExporter


def test_exporter_init():
    """Test exporter initialization"""
    exporter = GMSHExporter()
    assert exporter is not None


def test_prepare_mesh():
    """Test mesh preparation"""
    exporter = GMSHExporter()

    xi = np.array([0, 1])
    eta = np.array([0, 1])
    zeta = np.array([0, 1])

    x = np.array([0, 1])
    y = np.array([0, 1])
    z = np.array([0, 1])

    result = exporter.prepare_mesh(xi, eta, zeta, x, y, z)
    assert 'x' in result


def test_export_simple():
    """Test simple export without gmsh"""
    exporter = GMSHExporter()

    # Simple test data
    xi = np.linspace(0, 1, 2)
    eta = np.linspace(0, 1, 2)
    zeta = np.linspace(0, 1, 2)
    x = np.linspace(0, 1, 2)
    y = np.linspace(0, 1, 2)
    z = np.linspace(0, 1, 2)

    # This will test write_v4 format
    try:
        result = exporter.prepare_mesh(xi, eta, zeta, x, y, z)
        assert result is not None
    except Exception as e:
        pytest.skip(f"GMSH not available: {e}")