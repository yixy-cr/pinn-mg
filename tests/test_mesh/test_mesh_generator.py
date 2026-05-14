import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/mesh')
from mesh_generator import MeshGenerator


def test_mesh_generator_init():
    """Test mesh generator initialization"""
    gen = MeshGenerator(nx=10, ny=10, nz=10)
    assert gen.nx == 10
    assert gen.ny == 10
    assert gen.nz == 10


def test_generate():
    """Test hexahedral mesh generation"""
    gen = MeshGenerator(nx=5, ny=5, nz=5)
    xi, eta, zeta = gen.generate()

    assert len(xi) == 5
    assert len(eta) == 5
    assert len(zeta) == 5


def test_generate_flat():
    """Test flattened mesh generation"""
    gen = MeshGenerator(nx=5, ny=5, nz=5)
    xi, eta, zeta = gen.generate_flat()

    # Total points = 5 * 5 * 5 = 125
    assert len(xi) == 125
    assert len(eta) == 125
    assert len(zeta) == 125


def test_node_count():
    """Test node count"""
    gen = MeshGenerator(nx=10, ny=10, nz=10)
    assert gen.get_node_count() == 1000


def test_element_count():
    """Test element count"""
    gen = MeshGenerator(nx=10, ny=10, nz=10)
    # Hex elements: (nx-1) * (ny-1) * (nz-1)
    assert gen.get_element_count() == 729