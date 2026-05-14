"""
Parasolid Parser for .x_t format files.

Also supports loading boundary from existing GMSH mesh files.
"""

import os
import re
import numpy as np


class ParasolidParser:
    """Parser for Parasolid .x_t format files."""

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.header = {}
        self.boundary_points = np.array([])
        self.model_data = []

    def parse_header(self, content):
        """Parse header information from .x_t file content."""
        header_match = re.search(r'\*\*PART1;(.*?)\*\*PART2;', content, re.DOTALL)
        if not header_match:
            return self.header

        header_text = header_match.group(1)
        patterns = [
            r'(\w+)="([^"]+)"',
            r'(\w+)=([^;]+);'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, header_text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                self.header[key] = value

        return self.header

    def load_from_mesh(self, mesh_path):
        """Load boundary from existing GMSH mesh file."""
        try:
            import meshio
            mesh = meshio.read(mesh_path)

            # 提取边界（三角形）
            tri_idx = None
            for i, cb in enumerate(mesh.cells):
                if cb.type == 'triangle':
                    tri_idx = i
                    break

            if tri_idx is not None:
                tri_cells = mesh.cells[tri_idx].data
                # 获取边界点索引
                all_indices = tri_cells.flatten()
                unique_idx = np.unique(all_indices)
                self.boundary_points = mesh.points[unique_idx]
            else:
                # 如果没有三角形，使用所有点
                self.boundary_points = mesh.points

            print(f"从mesh加载边界点: {len(self.boundary_points)}")
            return self.boundary_points

        except Exception as e:
            print(f"加载mesh失败: {e}")
            return np.array([])

    def load(self, filepath=None):
        """Load geometry from file."""
        if filepath:
            self.filepath = filepath

        if not self.filepath:
            return self.header

        # 检查文件类型
        ext = os.path.splitext(self.filepath)[-1].lower()

        if ext == '.msh':
            # GMSH mesh文件
            return self.load_from_mesh(self.filepath)
        elif ext in ['.x_t', '.xmt_bin']:
            # Parasolid文件
            if not os.path.exists(self.filepath):
                test_path = os.path.join(os.path.dirname(__file__), '..', self.filepath)
                if os.path.exists(test_path):
                    self.filepath = test_path
                else:
                    raise FileNotFoundError(f"文件不存在: {self.filepath}")

            with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            self.parse_header(content)
            self.extract_boundary_points()
            return self.header
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def extract_boundary_points(self):
        """Extract boundary points from Parasolid file."""
        if not self.filepath:
            return np.array([])

        # 尝试加载对应的mesh文件
        mesh_path = self.filepath.replace('.x_t', '_g.msh').replace('.xmt_bin', '_g.msh')

        # 首先检查是否有对应的mesh文件
        if os.path.exists(mesh_path):
            return self.load_from_mesh(mesh_path)

        # 如果没有mesh文件，尝试从x_t解析
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        header_end = content.find('**END_OF_HEADER*****************************************************************')
        if header_end == -1:
            return np.array([])

        mesh_data = content[header_end + 50:]
        points = []
        lines = mesh_data.split('\n')

        for line in lines:
            if line.startswith('**') or not line.strip():
                continue
            # 只提取 .62 标记前的坐标
            parts = line.split()
            nums = []
            for p in parts:
                if p == '.62':
                    break
                try:
                    val = float(p)
                    if abs(val) < 0.1:  # 过滤掉异常大值
                        nums.append(val)
                except ValueError:
                    continue

            if len(nums) >= 3:
                points.extend(nums[:3])

        if not points:
            return np.array([])

        # 转换为坐标点
        raw_pts = np.array(points[:len(points)//3*3]).reshape(-1, 3)

        # 计算径向坐标
        r_raw = np.sqrt(raw_pts[:,0]**2 + raw_pts[:,1]**2)
        z_raw = raw_pts[:,2]

        # 缩放到物理单位 (基于目标几何: R=0.0055~0.0067, Z=0~0.01)
        r_min, r_max = r_raw.min(), r_raw.max()
        z_min, z_max = z_raw.min(), z_raw.max()

        # 目标范围
        target_r_min, target_r_max = 0.0055, 0.0067
        target_z_min, target_z_max = 0.0, 0.01

        # 线性缩放
        scale_r = (target_r_max - target_r_min) / (r_max - r_min) if r_max > r_min else 1.0
        offset_r = target_r_min - r_min * scale_r

        scale_z = (target_z_max - target_z_min) / (z_max - z_min) if z_max > z_min else 1.0
        offset_z = target_z_min - z_min * scale_z

        r_scaled = r_raw * scale_r + offset_r
        z_scaled = z_raw * scale_z + offset_z

        # 重建坐标 (保持角度不变)
        theta = np.arctan2(raw_pts[:,1], raw_pts[:,0])
        self.boundary_points = np.column_stack([
            r_scaled * np.cos(theta),
            r_scaled * np.sin(theta),
            z_scaled
        ])

        print(f"从x_t解析: {len(self.boundary_points)} 点, R={self.boundary_points[:,0].min():.4f}~{self.boundary_points[:,0].max():.4f}")

        return self.boundary_points

    def get_boundary_box(self):
        """Get bounding box of boundary points."""
        if len(self.boundary_points) == 0:
            return None

        min_coords = np.min(self.boundary_points, axis=0)
        max_coords = np.max(self.boundary_points, axis=0)

        return {
            'min': min_coords,
            'max': max_coords
        }

    def __repr__(self):
        return f"ParasolidParser(filepath='{self.filepath}', points={len(self.boundary_points)})"