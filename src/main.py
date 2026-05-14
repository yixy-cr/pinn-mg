#!/usr/bin/env python3
"""
PINN-MG: Physics-Informed Neural Networks for Mesh Generation.
Main entry point.
"""

import argparse
import os
import sys

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import yaml
import torch
import numpy as np

from geometry.parasolid_parser import ParasolidParser
from geometry.boundary_fitter import BoundaryFitter
from network.pinn_network import PINNNetwork
from physics.loss_function import PINNLoss
from mesh.mesh_generator import MeshGenerator
from mesh.gmsh_exporter import GMSHExporter


def load_config(config_path):
    """Load configuration."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(model, loss_fn, xi, eta, zeta, x_bc, y_bc, z_bc, config, device='cpu'):
    """Training loop."""
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['decay_interval'],
        gamma=config['training']['lr_decay']
    )

    epochs = config['training']['epochs']
    model.train()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(xi, eta, zeta)
        x, y, z = output[0], output[1], output[2]

        # Compute loss
        loss = loss_fn(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)
    if not config:
        config = {
            'model': {'hidden_dim': 64, 'num_layers': 4, 'activation': 'tanh'},
            'training': {'epochs': 15000, 'learning_rate': 1e-5, 'lr_decay': 0.99, 'decay_interval': 1000},
            'physics': {'lambda': 1.0, 'mu': 0.35},
            'mesh': {'nx': 10, 'ny': 10, 'nz': 10},
            'boundary': {'svr_C': 1.0, 'svr_epsilon': 0.01}
        }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Geometry
    if args.input:
        print(f"Loading geometry from {args.input}...")
        parser = ParasolidParser(args.input)
        parser.load(args.input)
        boundary_points = parser.boundary_points
        print(f"Loaded {len(boundary_points)} boundary points")
    else:
        print("No input geometry, using default boundary")
        boundary_points = None

    # Fit boundary - use direct linear mapping (参数→物理是线性关系，不需要SVR)
    if boundary_points is not None and len(boundary_points) > 0:
        # Get boundary bounds
        x_min, x_max = boundary_points[:, 0].min(), boundary_points[:, 0].max()
        y_min, y_max = boundary_points[:, 1].min(), boundary_points[:, 1].max()
        z_min, z_max = boundary_points[:, 2].min(), boundary_points[:, 2].max()

        # 分析边界形状 - 计算径向范围
        r = np.sqrt(boundary_points[:,0]**2 + boundary_points[:,1]**2)
        r_min, r_max = r.min(), r.max()

        print(f"边界范围: r=[{r_min:.4f},{r_max:.4f}], z=[{z_min:.4f},{z_max:.4f}]")
    else:
        r_min, r_max = 0.0055, 0.0067
        z_min, z_max = 0.0, 0.01

    # === PINN训练模式 ===
    # 创建参数化网格输入 (xi, eta, zeta) ∈ [0,1]³
    nx = config['mesh']['nx']
    ny = config['mesh']['ny']
    nz = config['mesh']['nz']

    xi = np.linspace(0, 1, nx)
    eta = np.linspace(0, 1, ny)
    zeta = np.linspace(0, 1, nz)

    xi_m, eta_m, zeta_m = np.meshgrid(xi, eta, zeta, indexing='ij')
    xi_f = xi_m.flatten()  # rho归一化半径
    eta_f = eta_m.flatten()  # theta角度
    zeta_f = zeta_m.flatten()  # zeta Z向

    # 边界条件：参数->物理的极坐标映射
    # xi (rho) -> r, eta (theta) -> angle, zeta -> z
    # 简单线性映射，避免极坐标
    x_bc = r_min + xi_f * (r_max - r_min)
    y_bc = r_min + xi_f * (r_max - r_min)  # 简化
    z_bc = z_min + zeta_f * (z_max - z_min)

    # === 目标边界条件：极坐标 ===
    # xi -> r, eta -> theta, zeta -> z
    r = r_min + xi_f * (r_max - r_min)
    x_target = r * np.cos(2 * np.pi * eta_f)
    y_target = r * np.sin(2 * np.pi * eta_f)
    z_target = z_min + zeta_f * (z_max - z_min)

    # === 简化解析解（非极坐标）===
    # 简化为线性映射：x = r, y = r, z = z
    x_simple = r
    y_simple = r
    z_simple = z_min + zeta_f * (z_max - z_min)

    # 残差 = 目标 - 简化
    x_residual = x_target - x_simple
    y_residual = y_target - y_simple
    z_residual = z_target - z_simple

    print(f"残差范围: x={x_residual.min():.6f}~{x_residual.max():.6f}, y={y_residual.min():.6f}~{y_residual.max():.6f}")

    # === PINN网络训练 ===
    if args.train:
        # 初始化网络
        print("Initializing PINN network...")
        model = PINNNetwork(
            input_dim=3,
            hidden_dim=config['model']['hidden_dim'],
            output_dim=3,
            num_layers=config['model']['num_layers'],
            activation=config['model']['activation']
        )

        # 转换为tensor
        xi_t = torch.tensor(xi_f, dtype=torch.float32, requires_grad=True)
        eta_t = torch.tensor(eta_f, dtype=torch.float32, requires_grad=True)
        zeta_t = torch.tensor(zeta_f, dtype=torch.float32, requires_grad=True)

        # 残差目标
        x_residual = torch.tensor(x_residual, dtype=torch.float32)
        y_residual = torch.tensor(y_residual, dtype=torch.float32)
        z_residual = torch.tensor(z_residual, dtype=torch.float32)

        # 训练
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        epochs = config['training']['epochs']
        model.train()

        print(f"Training PINN for {epochs} epochs...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            output = model(xi_t, eta_t, zeta_t)
            x_res, y_res, z_res = output[0], output[1], output[2]

            # 残差损失
            loss = torch.mean((x_res - x_residual)**2 + (y_res - y_residual)**2 + (z_res - z_residual)**2)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # 生成输出: 简化解 + 残差
        print("Generating output mesh...")
        model.eval()
        with torch.no_grad():
            residual = model(xi_t, eta_t, zeta_t)
            x_out = x_simple + residual[0].numpy()
            y_out = y_simple + residual[1].numpy()
            z_out = z_simple + residual[2].numpy()
    else:
        # 不训练时使用解析解
        x_out = x_bc
        y_out = y_bc
        z_out = z_bc

    print(f"Generated {len(x_out)} mesh points")

    # Export
    output_path = args.output or 'output.msh'
    print(f"Exporting to {output_path}...")
    exporter = GMSHExporter()
    exporter.export(xi_f, eta_f, zeta_f, x_out, y_out, z_out, output_path)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN-MG')
    parser.add_argument('--input', type=str, default='', help='Input .x_t file')
    parser.add_argument('--output', type=str, default='output.msh', help='Output .msh file')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    parser.add_argument('--train', action='store_true', help='Training mode')

    args = parser.parse_args()
    main(args)