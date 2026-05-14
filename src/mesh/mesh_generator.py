"""
Mesh Generator for Parametric Hexahedral Mesh.

Generates hexahedral mesh in computational domain (xi, eta, zeta).
"""

import numpy as np


class MeshGenerator:
    """Generate parametric domain hexahedral mesh."""

    def __init__(self, nx=20, ny=20, nz=20,
                 xi_range=(0, 1), eta_range=(0, 1), zeta_range=(0, 1)):
        """
        Initialize mesh generator.

        Args:
            nx: Number of points in xi direction
            ny: Number of points in eta direction
            nz: Number of points in zeta direction
            xi_range: (min, max) for xi
            eta_range: (min, max) for eta
            zeta_range: (min, max) for zeta
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xi_range = xi_range
        self.eta_range = eta_range
        self.zeta_range = zeta_range

    def generate(self):
        """
        Generate hexahedral mesh in parametric domain.

        Returns:
            xi, eta, zeta: 1D arrays (not meshed)
        """
        xi = np.linspace(self.xi_range[0], self.xi_range[1], self.nx)
        eta = np.linspace(self.eta_range[0], self.eta_range[1], self.ny)
        zeta = np.linspace(self.zeta_range[0], self.zeta_range[1], self.nz)

        return xi, eta, zeta

    def generate_flat(self):
        """
        Generate flattened mesh arrays.

        Returns:
            xi_flat, eta_flat, zeta_flat: 1D arrays (all points)
        """
        xi, eta, zeta = self.generate()

        # Create 3D meshgrid
        xi_mesh, eta_mesh, zeta_mesh = np.meshgrid(xi, eta, zeta, indexing='ij')

        # Flatten to 1D
        xi_flat = xi_mesh.flatten()
        eta_flat = eta_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()

        return xi_flat, eta_flat, zeta_flat

    def generate_meshgrid(self):
        """
        Generate 3D meshgrid arrays.

        Returns:
            xi_mesh, eta_mesh, zeta_mesh: 3D arrays
        """
        xi, eta, zeta = self.generate()
        xi_mesh, eta_mesh, zeta_mesh = np.meshgrid(xi, eta, zeta, indexing='ij')

        return xi_mesh, eta_mesh, zeta_mesh

    def get_node_count(self):
        """Get total node count."""
        return self.nx * self.ny * self.nz

    def get_element_count(self):
        """Get hexahedral element count."""
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)

    def get_element_connectivity(self):
        """
        Get hexahedral element connectivity (node indices).

        Returns:
            connectivity: Array of shape (num_elements, 8) containing node indices
        """
        elements = []

        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                for k in range(self.nz - 1):
                    # Node indices for hex element
                    n1 = i * self.ny * self.nz + j * self.nz + k
                    n2 = (i + 1) * self.ny * self.nz + j * self.nz + k
                    n3 = (i + 1) * self.ny * self.nz + (j + 1) * self.nz + k
                    n4 = i * self.ny * self.nz + (j + 1) * self.nz + k
                    n5 = i * self.ny * self.nz + j * self.nz + (k + 1)
                    n6 = (i + 1) * self.ny * self.nz + j * self.nz + (k + 1)
                    n7 = (i + 1) * self.ny * self.nz + (j + 1) * self.nz + (k + 1)
                    n8 = i * self.ny * self.nz + (j + 1) * self.nz + (k + 1)

                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

        return np.array(elements)

    def __repr__(self):
        return f"MeshGenerator(nx={self.nx}, ny={self.ny}, nz={self.nz})"