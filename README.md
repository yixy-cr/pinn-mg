# PINN-MG

Physics-Informed Neural Networks for Mesh Generation.

## Overview

PINN-MG implements the PINN-MG algorithm to generate hexahedral meshes from CAD geometry files. The method uses Physics-Informed Neural Networks (PINN) with the 3D Navier-Lamé equation as a physical constraint.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py --input data/1pinflue4wire.x_t --output output/mesh.msh
```

## Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (hidden_dim, num_layers)
- Training parameters (epochs, learning_rate)
- Physics constants (lambda, mu)
- Mesh resolution (nx, ny, nz)

## Architecture

- **Input:** Parasolid .x_t CAD geometry
- **Network:** PINN with input enhancement (tan/cot)
- **Constraint:** 3D Navier-Lamé equation
- **Output:** GMSH v4 hexahedral mesh

## References

- PINN-MG: Physics-Informed Neural Networks with Multigrid Solver (paper)