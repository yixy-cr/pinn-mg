"""
GMSH Exporter for Hexahedral Mesh.

Uses meshio to generate GMSH-compatible files.
"""

import numpy as np
import meshio


class GMSHExporter:
    """Export hexahedral mesh to GMSH format."""

    def __init__(self):
        pass

    def export(self, xi, eta, zeta, x_pred, y_pred, z_pred, output_path):
        """
        Export to GMSH format.
        """
        x_flat = x_pred.flatten() if hasattr(x_pred, 'flatten') else np.array(x_pred)
        y_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else np.array(y_pred)
        z_flat = z_pred.flatten() if hasattr(z_pred, 'flatten') else np.array(z_pred)

        n = len(x_flat)
        points = np.column_stack([x_flat, y_flat, z_flat]).astype(np.float64)

        # 尝试恢复原始网格尺寸
        # nx*ny*nz = n
        # 尝试常见组合
        options = [
            (8, 16, 32),  # 4096 default
            (16, 16, 32), # 8192
            (16, 32, 32),# 16384
            (5, 20, 20), # 2000
        ]

        nx = ny = nz = 8  # default
        found = False
        for nx_, ny_, nz_ in options:
            if nx_ * ny_ * nz_ == n:
                nx, ny, nz = nx_, ny_, nz_
                found = True
                break

        if not found:
            nx = ny = nz = int(round(n ** (1/3)))

        print(f"Creating hex mesh: {nx}x{ny}x{nz}")

        # 创建六面体元素
        hex_cells = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    n1 = i * ny * nz + j * nz + k
                    n2 = (i + 1) * ny * nz + j * nz + k
                    n3 = (i + 1) * ny * nz + (j + 1) * nz + k
                    n4 = i * ny * nz + (j + 1) * nz + k
                    n5 = i * ny * nz + j * nz + (k + 1)
                    n6 = (i + 1) * ny * nz + j * nz + (k + 1)
                    n7 = (i + 1) * ny * nz + (j + 1) * nz + (k + 1)
                    n8 = i * ny * nz + (j + 1) * nz + (k + 1)
                    hex_cells.append([n1, n2, n3, n4, n5, n6, n7, n8])

        cells = [('hexahedron', np.array(hex_cells))]
        print(f"Created {len(hex_cells)} hex elements")

        mesh = meshio.Mesh(points, cells)
        mesh.write(output_path, file_format='gmsh')
        print(f"Exported to {output_path}")