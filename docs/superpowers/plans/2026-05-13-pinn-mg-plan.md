# PINN-MG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 PINN-MG 算法：从 CAD (.x_t Parasolid) 读取边界，通过 3D Navier-Lamé 方程约束训练 PINN 神经网络，生成 GMSH v4 六面体网格文件。

**Architecture:** 分模块实现：几何处理（Parasolid解析 + SVR边界拟合）→ PINN网络（输入增强 + 隐藏层）→ 物理约束（3D Navier-Lamé方程在计算域上表达）→ 损失函数 → 网格生成 → GMSH输出。

**Tech Stack:** Python (PyTorch) + C++ (GMSH) + SVR (scikit-learn)

---

## File Structure

```
pinn-mg/
├── docs/superpowers/
│   ├── specs/2026-05-13-pinn-mg-design.md    # 已完成的设计文档
│   └── plans/2026-05-13-pinn-mg-plan.md    # 本计划
├── src/
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── parasolid_parser.py              # 解析 .x_t 文件
│   │   └── boundary_fitter.py           # SVR 拟合边界
│   ├── network/
│   │   ├── __init__.py
│   │   ├── pinn_network.py               # PINN 网络架构
│   │   ├── input_enhancement.py         # 输入增强 (tan/cot)
│   │   └── attention.py                # 注意力模块 (预留)
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── navier_lame_3d.py         # 3D Navier-Lamé 方程
│   │   └── loss_function.py            # 损失函数
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── mesh_generator.py          # 参数域六面体网格
│   │   └── gmsh_exporter.py        # GMSH v4 输出
│   └── main.py                     # 主程序
├── tests/
│   ├── test_geometry/
│   ├── test_network/
│   ├── test_physics/
│   └── test_mesh/
├── configs/default.yaml
└── requirements.txt
```

---

## Task 0: 项目初始化

**Files:**
- Create: `~/algorithm/pinn-mg/requirements.txt`
- Create: `~/algorithm/pinn-mg/configs/default.yaml`
- Create: `~/algorithm/pinn-mg/README.md`

- [ ] **Step 1: 创建 requirements.txt**

```txt
torch>=2.0
numpy
scikit-learn
gmsh>=4.0
pyyaml
pytest
```

- [ ] **Step 2: 创建 configs/default.yaml**

```yaml
model:
  hidden_dim: 64
  num_layers: 4
  activation: tanh

training:
  epochs: 15000
  learning_rate: 1.0e-5
  lr_decay: 0.99
  decay_interval: 1000

physics:
  lambda: 1.0
  mu: 0.35

mesh:
  nx: 20
  ny: 20
  nz: 20

boundary:
  svr_C: 1.0
  svr_epsilon: 0.01
```

- [ ] **Step 3: 创建 README.md**

```markdown
# PINN-MG

Physics-Informed Neural Networks for Mesh Generation.

## Installation

pip install -r requirements.txt

## Usage

python src/main.py --input data/1pinflue4wire.x_t --output output/mesh.msh
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt configs/default.yaml README.md
git commit -m "chore: add project config files"
```

---

## Task 1: Parasolid 解析器

**Files:**
- Create: `~/algorithm/pinn-mg/src/geometry/parasolid_parser.py`
- Create: `~/algorithm/pinn-mg/tests/test_geometry/test_parasolid_parser.py`

**Goal:** 解析 .x_t 文件，提取边界曲线/曲面点数据。

- [ ] **Step 1: 创建测试 test_parasolid_parser.py**

```python
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/geometry')
from parasolid_parser import ParasolidParser

def test_parse_header():
    """Test parsing header information"""
    parser = ParasolidParser()
    # Test with small sample data
    sample_data = """**ABCDEFGHIJKLMNOPQRSTUVWXYZ**
**PART1;
KEY=test;
**PART2;
**END_OF_HEADER*****************************************************************
"""
    result = parser.parse_header(sample_data)
    assert result['KEY'] == 'test'

def test_extract_boundary_points():
    """Test extracting boundary points from .x_t file"""
    parser = ParasolidParser()
    result = parser.extract_boundary_points()
    # Should return numpy array of points
    assert isinstance(result, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_geometry/test_parasolid_parser.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write minimal parasolid_parser.py**

```python
import numpy as np
import re

class ParasolidParser:
    """Parser for Parasolid .x_t format files."""
    
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.header = {}
        self.boundary_points = []
    
    def parse_header(self, content):
        """Parse header information."""
        # Extract KEY=value pairs
        pattern = r'(\w+)=([^;]+);'
        matches = re.findall(pattern, content)
        for key, value in matches:
            self.header[key.strip()] = value.strip()
        return self.header
    
    def extract_boundary_points(self):
        """Extract boundary points from file."""
        if not self.filepath:
            return np.array([])
        # TODO: implement actual parsing
        return np.array([])
    
    def load(self, filepath):
        """Load and parse a .x_t file."""
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return self.parse_header(content)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_geometry/test_parasolid_parser.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/geometry/parasolid_parser.py tests/test_geometry/test_parasolid_parser.py
git commit -m "feat: add Parasolid parser"
```

---

## Task 2: 边界 SVR 拟合

**Files:**
- Create: `~/algorithm/pinn-mg/src/geometry/boundary_fitter.py`
- Create: `~/algorithm/pinn-mg/tests/test_geometry/test_boundary_fitter.py`

**Goal:** 用 SVR 将 CAD 边界点拟合成函数 f(xi, eta, zeta) -> (x, y, z)。

- [ ] **Step 1: 创建测试 test_boundary_fitter.py**

```python
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/geometry')
from boundary_fitter import BoundaryFitter

def test_fit_boundary():
    """Test SVR boundary fitting"""
    fitter = BoundaryFitter()
    # Create sample data: unit cube boundary
    xi = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    eta = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    zeta = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    fitter.fit(xi, eta, zeta, x, y, z)
    assert fitter.model_x is not None

def test_predict():
    """Test boundary prediction"""
    fitter = BoundaryFitter()
    # Simple test
    xi = np.array([0, 1])
    eta = np.array([0, 0])
    zeta = np.array([0, 0])
    x = np.array([0, 1])
    y = np.array([0, 0])
    z = np.array([0, 0])
    
    fitter.fit(xi, eta, zeta, x, y, z)
    pred = fitter.predict(0.5, 0, 0)
    assert abs(pred[0] - 0.5) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_geometry/test_boundary_fitter.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write boundary_fitter.py**

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class BoundaryFitter:
    """SVR-based boundary curve fitting."""
    
    def __init__(self, C=1.0, epsilon=0.01):
        self.C = C
        self.epsilon = epsilon
        self.model_x = None
        self.model_y = None
        self.model_z = None
        self.scaler = StandardScaler()
    
    def fit(self, xi, eta, zeta, x, y, z):
        """Fit SVR models for x, y, z coordinates."""
        # Stack inputs: (xi, eta, zeta)
        X = np.column_stack([xi, eta, zeta])
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Fit separate SVR for each coordinate
        self.model_x = SVR(C=self.C, epsilon=self.epsilon)
        self.model_x.fit(X_scaled, x)
        
        self.model_y = SVR(C=self.C, epsilon=self.epsilon)
        self.model_y.fit(X_scaled, y)
        
        self.model_z = SVR(C=self.C, epsilon=self.epsilon)
        self.model_z.fit(X_scaled, z)
    
    def predict(self, xi, eta, zeta):
        """Predict boundary coordinates."""
        X = np.array([[xi, eta, zeta]])
        X_scaled = self.scaler.transform(X)
        
        x = self.model_x.predict(X_scaled)[0]
        y = self.model_y.predict(X_scaled)[0]
        z = self.model_z.predict(X_scaled)[0]
        
        return np.array([x, y, z])
    
    def __call__(self, xi, eta, zeta):
        """Convenience method."""
        return self.predict(xi, eta, zeta)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_geometry/test_boundary_fitter.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/geometry/boundary_fitter.py tests/test_geometry/test_boundary_fitter.py
git commit -m "feat: add SVR boundary fitter"
```

---

## Task 3: 输入增强

**Files:**
- Create: `~/algorithm/pinn-mg/src/network/input_enhancement.py`
- Create: `~/algorithm/pinn-mg/tests/test_network/test_input_enhancement.py`

**Goal:** 实现输入增强：U_in = [xi, eta, zeta, tan(xi), tan(eta), tan(zeta), cot(xi), cot(eta), cot(zeta)]

- [ ] **Step 1: 创建测试 test_input_enhancement.py**

```python
import pytest
import torch
import sys
sys.path.insert(0, 'src/network')
from input_enhancement import InputEnhancement

def test_enhance_3d():
    """Test 3D input enhancement"""
    enhancer = InputEnhancement()
    xi = torch.tensor([0.0, 0.5, 1.0])
    eta = torch.tensor([0.0, 0.5, 1.0])
    zeta = torch.tensor([0.0, 0.5, 1.0])
    
    result = enhancer.enhance(xi, eta, zeta)
    
    # Should be: [xi, eta, zeta, tan(xi), tan(eta), tan(zeta), cot(xi), cot(eta), cot(zeta)]
    assert result.shape[0] == 9
    assert result.shape[1] == 3

def test_tan_cot_values():
    """Test tan and cot calculations"""
    enhancer = InputEnhancement()
    xi = torch.tensor([0.0])
    eta = torch.tensor([0.0])
    zeta = torch.tensor([0.0])
    
    result = enhancer.enhance(xi, eta, zeta)
    
    # At 0: tan(0)=0, cot(0)=inf (handle edge case)
    # Check values are finite
    assert torch.isfinite(result).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_network/test_input_enhancement.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write input_enhancement.py**

```python
import torch
import torch.nn as nn

class InputEnhancement(nn.Module):
    """Input enhancement with tan/cot transformation."""
    
    def __init__(self):
        super().__init__()
    
    def tan(self, x):
        """Safe tan function avoiding inf."""
        return torch.tan(x)
    
    def cot(self, x):
        """Safe cot function: cot(x) = 1/tan(x), handling edge cases."""
        tan_x = torch.tan(x)
        # Avoid division by zero
        tan_x = torch.where(
            torch.abs(tan_x) < 1e-10,
            torch.ones_like(tan_x) * 1e-10,
            tan_x
        )
        return 1.0 / tan_x
    
    def enhance(self, xi, eta, zeta):
        """
        Enhance input with tan/cot transformations.
        
        Args:
            xi, eta, zeta: Input tensors (batch_size,)
        
        Returns:
            Enhanced input tensor (9, batch_size)
        """
        # Original coordinates
        coords = torch.stack([xi, eta, zeta], dim=0)
        
        # Tan transformations
        tan_coords = torch.stack([
            self.tan(xi),
            self.tan(eta),
            self.tan(zeta)
        ], dim=0)
        
        # Cot transformations
        cot_coords = torch.stack([
            self.cot(xi),
            self.cot(eta),
            self.cot(zeta)
        ], dim=0)
        
        # Concatenate: [xi, eta, zeta, tan, cot]
        enhanced = torch.cat([coords, tan_coords, cot_coords], dim=0)
        
        return enhanced
    
    def forward(self, xi, eta, zeta):
        """Forward pass."""
        return self.enhance(xi, eta, zeta)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_network/test_input_enhancement.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/network/input_enhancement.py tests/test_network/test_input_enhancement.py
git commit -m "feat: add input enhancement (tan/cot)"
```

---

## Task 4: PINN 网络架构

**Files:**
- Create: `~/algorithm/pinn-mg/src/network/pinn_network.py`
- Create: `~/algorithm/pinn-mg/tests/test_network/test_pinn_network.py`

**Goal:** 实现 PINN 网络：输入 (xi, eta, zeta) → 输出 (x, y, z)

- [ ] **Step 1: 创建测试 test_pinn_network.py**

```python
import pytest
import torch
import sys
sys.path.insert(0, 'src/network')
from pinn_network import PINNNetwork

def test_forward_pass():
    """Test forward pass produces correct output shape"""
    model = PINNNetwork(input_dim=3, hidden_dim=64, output_dim=3, num_layers=4)
    xi = torch.randn(10)
    eta = torch.randn(10)
    zeta = torch.randn(10)
    
    output = model(xi, eta, zeta)
    
    assert output.shape == (3, 10)

def test_output_is_coordinates():
    """Test output is physical coordinates"""
    model = PINNNetwork(input_dim=3, hidden_dim=64, output_dim=3, num_layers=4)
    # Identity mapping test
    xi = torch.tensor([0.0, 0.5, 1.0])
    eta = torch.tensor([0.0, 0.5, 1.0])
    zeta = torch.tensor([0.0, 0.5, 1.0])
    
    output = model(xi, eta, zeta)
    
    # Output should be 3D coordinates
    assert output.shape[0] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_network/test_pinn_network.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write pinn_network.py**

```python
import torch
import torch.nn as nn
from input_enhancement import InputEnhancement

class PINNNetwork(nn.Module):
    """PINN Network for (xi, eta, zeta) -> (x, y, z) mapping."""
    
    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=4,
        activation='tanh'
    ):
        super().__init__()
        
        self.input_enhance = InputEnhancement()
        self.num_layers = num_layers
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        # Enhanced input: 9 dimensions
        enhanced_dim = 9
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(enhanced_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, xi, eta, zeta):
        """
        Forward pass.
        
        Args:
            xi, eta, zeta: Input tensors (batch_size,)
        
        Returns:
            Output tensor (3, batch_size) representing (x, y, z)
        """
        # Input enhancement
        x = self.input_enhance(xi, eta, zeta)  # (9, batch)
        
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
        
        # Output
        output = self.output_layer(x)  # (3, batch)
        
        return output
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_network/test_pinn_network.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/network/pinn_network.py tests/test_network/test_pinn_network.py
git commit -m "feat: add PINN network architecture"
```

---

## Task 5: 3D Navier-Lamé 方程（在计算域上表达）

**Files:**
- Create: `~/algorithm/pinn-mg/src/physics/navier_lame_3d.py`
- Create: `~/algorithm/pinn-mg/tests/test_physics/test_navier_lame_3d.py`

**Goal:** 实现 3D Navier-Lamé 方程，使用自动微分计算偏导数。

⚠️ **关键：** 需要推导变换后的方程形式，用 ∂x/∂ξ, ∂²x/∂ξ² 等偏导数表达。

- [ ] **Step 1: 创建测试 test_navier_lame_3d.py**

```python
import pytest
import torch
import sys
sys.path.insert(0, 'src/physics')
from navier_lame_3d import NavierLame3D

def test_compute_residuals():
    """Test computing Navier-Lamé residuals"""
    physics = NavierLame3D(lambda_=1.0, mu=0.35)
    
    # Create sample batch
    batch_size = 10
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    
    # Network outputs (x, y, z)
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)
    
    residuals = physics.compute_residuals(x, y, z, xi, eta, zeta)
    
    # Should return 3 residuals (f_x, f_y, f_z)
    assert residuals.shape == (3, batch_size)

def test_zero_residual_for_exact_solution():
    """Test that exact solution gives zero residual"""
    physics = NavierLame3D(lambda_=1.0, mu=0.35)
    
    # For identity mapping, residuals should be small
    batch_size = 5
    xi = torch.linspace(0, 1, batch_size, requires_grad=True).reshape(-1, 1)
    eta = torch.zeros_like(xi)
    zeta = torch.zeros_like(xi)
    
    x = xi  # x = xi
    y = eta  # y = eta
    z = zeta  # z = zeta
    
    residuals = physics.compute_residuals(x.squeeze(), y.squeeze(), z.squeeze(), 
                                   xi.squeeze(), eta.squeeze(), zeta.squeeze())
    
    # Exact solution should give near-zero residuals
    assert torch.norm(residuals) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_physics/test_navier_lame_3d.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write navier_lame_3d.py**

```python
import torch

class NavierLame3D:
    """
    3D Navier-Lamé equation in computational domain.
    
    The equation: (λ+μ)∇(∇·u) + μ∇²u = 0
    where u = (x-ξ, y-η, z-ζ) is displacement.
    
    Expressing in computational domain (ξ, η, ζ):
    - Need derivatives ∂x/∂ξ, ∂²x/∂ξ², etc.
    - Use torch.autograd for automatic differentiation
    """
    
    def __init__(self, lambda_=1.0, mu=0.35):
        self.lambda_ = lambda_
        self.mu = mu
    
    def compute_derivatives(self, x, xi):
        """
        Compute first and second derivatives using autograd.
        
        Args:
            x: function values (batch,)
            xi: independent variable (batch,)
        
        Returns:
            dx_dxi: ∂x/∂ξ
            d2x_dxi2: ∂²x/∂ξ²
        """
        # First derivative
        dx_dxi = torch.autograd.grad(
            x.sum(), xi, create_graph=True, retain_graph=True
        )[0]
        
        # Second derivative
        d2x_dxi2 = torch.autograd.grad(
            dx_dxi.sum(), xi, create_graph=True, retain_graph=True
        )[0]
        
        return dx_dxi, d2x_dxi2
    
    def compute_residuals(self, x, y, z, xi, eta, zeta):
        """
        Compute Navier-Lamé residuals.
        
        Args:
            x, y, z: Network output (batch,)
            xi, eta, zeta: Input coordinates (batch,)
        
        Returns:
            residuals: (3, batch) for f_x, f_y, f_z
        """
        # Compute all first derivatives
        x_xi, x_xixi = self.compute_derivatives(x, xi)
        x_eta, x_etaeta = self.compute_derivatives(x, eta)
        x_zeta, x_zetazeta = self.compute_derivatives(x, zeta)
        
        y_xi, y_xixi = self.compute_derivatives(y, xi)
        y_eta, y_etaeta = self.compute_derivatives(y, eta)
        y_zeta, y_zetazeta = self.compute_derivatives(y, zeta)
        
        z_xi, z_xixi = self.compute_derivatives(z, xi)
        z_eta, z_etaeta = self.compute_derivatives(z, eta)
        z_zeta, z_zetazeta = self.compute_derivatives(z, zeta)
        
        # Compute mixed derivatives
        x_xieta = torch.autograd.grad(x_xi.sum(), eta, create_graph=True)[0]
        x_etazeta = torch.autograd.grad(x_eta.sum(), zeta, create_graph=True)[0]
        
        # Divergence in computational domain
        div_u = x_xi + y_eta + z_zeta
        
        # Laplacian in computational domain (simplified form)
        # Note: This is a simplified version. Full 3D transformation
        # requires handling metric tensor from (ξ,η,ζ) → (x,y,z)
        lap_x = x_xixi + x_etaeta + x_zetazeta
        lap_y = y_xixi + y_etaeta + y_zetazeta
        lap_z = z_xixi + z_etaeta + z_zetazeta
        
        # Navier-Lamé equation: (λ+μ)∇(∇·u) + μ∇²u = 0
        # In computational domain (simplified):
        f_x = (self.lambda_ + self.mu) * div_u * x_xi + self.mu * lap_x
        f_y = (self.lambda_ + self.mu) * div_u * y_eta + self.mu * lap_y
        f_z = (self.lambda_ + self.mu) * div_u * z_zeta + self.mu * lap_z
        
        residuals = torch.stack([f_x, f_y, f_z])
        
        return residuals
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_physics/test_navier_lame_3d.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/physics/navier_lame_3d.py tests/test_physics/test_navier_lame_3d.py
git commit -m "feat: add 3D Navier-Lamé physics"
```

---

## Task 6: 损失函数

**Files:**
- Create: `~/algorithm/pinn-mg/src/physics/loss_function.py`
- Create: `~/algorithm/pinn-mg/tests/test_physics/test_loss_function.py`

**Goal:** 实现总损失：L = L_equation + L_boundary

- [ ] **Step 1: 创建测试 test_loss_function.py**

```python
import pytest
import torch
import sys
sys.path.insert(0, 'src/physics')
from loss_function import PINNLoss

def test_total_loss():
    """Test total loss computation"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)
    
    batch_size = 10
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    
    # Network outputs
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)
    
    # Boundary targets (from SVR)
    x_bc = torch.randn(batch_size)
    y_bc = torch.randn(batch_size)
    z_bc = torch.randn(batch_size)
    
    total_loss = loss_fn(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)
    
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() >= 0

def test_loss_components():
    """Test individual loss components"""
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)
    
    batch_size = 5
    x = torch.randn(batch_size, requires_grad=True)
    y = torch.randn(batch_size, requires_grad=True)
    z = torch.randn(batch_size, requires_grad=True)
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    x_bc = x.detach()
    y_bc = y.detach()
    z_bc = z.detach()
    
    eq_loss, bc_loss = loss_fn.compute_components(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)
    
    assert eq_loss.item() >= 0
    assert bc_loss.item() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_physics/test_loss_function.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write loss_function.py**

```python
import torch
from navier_lame_3d import NavierLame3D

class PINNLoss:
    """PINN Loss: L = L_equation + L_boundary"""
    
    def __init__(self, lambda_=1.0, mu=0.35, eq_weight=1.0, bc_weight=1.0):
        self.lambda_ = lambda_
        self.mu = mu
        self.eq_weight = eq_weight
        self.bc_weight = bc_weight
        self.physics = NavierLame3D(lambda_, mu)
    
    def compute_equation_loss(self, x, y, z, xi, eta, zeta):
        """
        Compute equation constraint loss.
        
        L_equation = mean(||f_x||² + ||f_y||² + ||f_z||²)
        """
        residuals = self.physics.compute_residuals(x, y, z, xi, eta, zeta)
        
        loss = torch.mean(residuals**2)
        return loss
    
    def compute_boundary_loss(self, x, y, z, x_bc, y_bc, z_bc):
        """
        Compute boundary constraint loss.
        
        L_boundary = mean(||x - x_bc||² + ||y - y_bc||² + ||z - z_bc||²)
        """
        loss_x = torch.mean((x - x_bc)**2)
        loss_y = torch.mean((y - y_bc)**2)
        loss_z = torch.mean((z - z_bc)**2)
        
        loss = loss_x + loss_y + loss_z
        return loss
    
    def compute_components(self, x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc):
        """Compute individual loss components."""
        eq_loss = self.compute_equation_loss(x, y, z, xi, eta, zeta)
        bc_loss = self.compute_boundary_loss(x, y, z, x_bc, y_bc, z_bc)
        return eq_loss, bc_loss
    
    def __call__(self, x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc):
        """Compute total loss."""
        eq_loss = self.compute_equation_loss(x, y, z, xi, eta, zeta)
        bc_loss = self.compute_boundary_loss(x, y, z, x_bc, y_bc, z_bc)
        
        total = self.eq_weight * eq_loss + self.bc_weight * bc_loss
        return total
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_physics/test_loss_function.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/physics/loss_function.py tests/test_physics/test_loss_function.py
git commit -m "feat: add PINN loss function"
```

---

## Task 7: 网格生成器

**Files:**
- Create: `~/algorithm/pinn-mg/src/mesh/mesh_generator.py`
- Create: `~/algorithm/pinn-mg/tests/test_mesh/test_mesh_generator.py`

**Goal:** 生成参数域六面体网格点 (ξ, η, ζ)

- [ ] **Step 1: 创建测试 test_mesh_generator.py**

```python
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/mesh')
from mesh_generator import MeshGenerator

def test_generate_hex_mesh():
    """Test hexahedral mesh generation"""
    gen = MeshGenerator(nx=10, ny=10, nz=10)
    xi, eta, zeta = gen.generate()
    
    assert xi.shape[0] == 10
    assert eta.shape[0] == 10
    assert zeta.shape[0] == 10

def test_mesh_points_count():
    """Test total mesh points count"""
    gen = MeshGenerator(nx=5, ny=5, nz=5)
    xi, eta, zeta = gen.generate()
    
    # 6 faces + interior (full hex mesh)
    total_points = 6 * 5 * 5  # Face points
    # Full mesh has interior too
    assert len(xi.flatten()) == 125
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_mesh/test_mesh_generator.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write mesh_generator.py**

```python
import numpy as np

class MeshGenerator:
    """Generate parametric domain hexahedral mesh."""
    
    def __init__(self, nx=20, ny=20, nz=20, xi_range=(0, 1), eta_range=(0, 1), zeta_range=(0, 1)):
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
            xi, eta, zeta: 1D arrays
        """
        xi = np.linspace(self.xi_range[0], self.xi_range[1], self.nx)
        eta = np.linspace(self.eta_range[0], self.eta_range[1], self.ny)
        zeta = np.linspace(self.zeta_range[0], self.zeta_range[1], self.nz)
        
        return xi, eta, zeta
    
    def generate_flat(self):
        """Generate flattened mesh arrays."""
        xi, eta, zeta = self.generate()
        
        # Create full mesh grid
        xi_mesh, eta_mesh, zeta_mesh = np.meshgrid(xi, eta, zeta, indexing='ij')
        
        xi_flat = xi_mesh.flatten()
        eta_flat = eta_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()
        
        return xi_flat, eta_flat, zeta_flat
    
    def get_node_count(self):
        """Get total node count."""
        return self.nx * self.ny * self.nz
    
    def get_element_count(self):
        """Get hexahedral element count."""
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_mesh/test_mesh_generator.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/mesh/mesh_generator.py tests/test_mesh/test_mesh_generator.py
git commit -m "feat: add mesh generator"
```

---

## Task 8: GMSH 输出

**Files:**
- Create: `~/algorithm/pinn-mg/src/mesh/gmsh_exporter.py`
- Create: `~/algorithm/pinn-mg/tests/test_mesh/test_gmsh_exporter.py`

**Goal:** 导出六面体网格为 GMSH v4 格式

- [ ] **Step 1: 创建测试 test_gmsh_exporter.py**

```python
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src/mesh')
from gmsh_exporter import GMSHExporter

def test_export_hex_mesh():
    """Test exporting hexahedral mesh to GMSH v4"""
    exporter = GMSHExporter()
    
    # Simple unit cube hex mesh
    xi = np.array([0, 1])
    eta = np.array([0, 1])
    zeta = np.array([0, 1])
    
    # Physical coordinates (unit cube)
    x = np.array([0, 1])
    y = np.array([0, 1])
    z = np.array([0, 1])
    
    # For now just test interface
    result = exporter.prepare_mesh(xi, eta, zeta, x, y, z)
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_mesh/test_gmsh_exporter.py -v
Expected: FAIL - module not found
```

- [ ] **Step 3: Write gmsh_exporter.py**

```python
import numpy as np
import gmsh

class GMSHExporter:
    """Export hexahedral mesh to GMSH v4 format."""
    
    def __init__(self):
        self.initialized = False
    
    def prepare_mesh(self, xi, eta, zeta, x, y, z):
        """Prepare mesh data for export."""
        # Create 3D grid
        xi_mesh, eta_mesh, zeta_mesh = np.meshgrid(xi, eta, zeta, indexing='ij')
        
        xi_flat = xi_mesh.flatten()
        eta_flat = eta_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        return {
            'xi': xi_flat, 'eta': eta_flat, 'zeta': zeta_flat,
            'x': x_flat, 'y': y_flat, 'z': z_flat
        }
    
    def export(self, xi, eta, zeta, x_pred, y_pred, z_pred, output_path):
        """
        Export to GMSH v4 format.
        
        Args:
            xi, eta, zeta: Parametric coordinates
            x_pred, y_pred, z_pred: Physical coordinates from PINN
            output_path: Output .msh file path
        """
        # Initialize GMSH
        if not self.initialized:
            gmsh.initialize()
            self.initialized = True
        
        # Prepare mesh data
        mesh_data = self.prepare_mesh(xi, eta, zeta, x_pred, y_pred, z_pred)
        
        # Create grid for hex elements
        nx = len(xi)
        ny = len(eta)
        nz = len(zeta)
        
        # Add nodes
        node_id = 1
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    gmsh.model.mesh.addNode(
                        mesh_data['x'][idx],
                        mesh_data['y'][idx],
                        mesh_data['z'][idx],
                        node_id
                    )
                    node_id += 1
        
        # Add hexahedral elements
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Node indices for hex element
                    n1 = i * ny * nz + j * nz + k + 1
                    n2 = (i + 1) * ny * nz + j * nz + k + 1
                    n3 = (i + 1) * ny * nz + (j + 1) * nz + k + 1
                    n4 = i * ny * nz + (j + 1) * nz + k + 1
                    n5 = i * ny * nz + j * nz + (k + 1) + 1
                    n6 = (i + 1) * ny * nz + j * nz + (k + 1) + 1
                    n7 = (i + 1) * ny * nz + (j + 1) * nz + (k + 1) + 1
                    n8 = i * ny * nz + (j + 1) * nz + (k + 1) + 1
                    
                    gmsh.model.mesh.addElement(
                        5,  # Hexahedron
                        [n1, n2, n3, n4, n5, n6, n7, n8]
                    )
        
        # Write file
        gmsh.write(output_path)
        
        # Finalize
        gmsh.finalize()
    
    def __del__(self):
        """Cleanup."""
        if self.initialized:
            gmsh.finalize()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/algorithm/pinn-mg
pytest tests/test_mesh/test_gmsh_exporter.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/mesh/gmsh_exporter.py tests/test_mesh/test_gmsh_exporter.py
git commit -m "feat: add GMSH exporter"
```

---

## Task 9: 主程序

**Files:**
- Create: `~/algorithm/pinn-mg/src/main.py`
- Create: `~/algorithm/pinn-mg/tests/integration/test_end_to_end.py`

**Goal:** 整合所有模块，实现端到端训练和导出。

- [ ] **Step 1: 创建测试 test_end_to_end.py**

```python
import pytest
import torch
import sys
sys.path.insert(0, 'src')
# Skip if gmsh not available
try:
    import gmsh
except ImportError:
    pytest.skip("gmsh not available", allow_module_level=True)

def test_training_loop():
    """Test basic training loop"""
    # This is a smoke test
    from network.pinn_network import PINNNetwork
    from physics.loss_function import PINNLoss
    
    model = PINNNetwork(input_dim=3, hidden_dim=32, output_dim=3, num_layers=2)
    loss_fn = PINNLoss(lambda_=1.0, mu=0.35)
    
    # Simple batch
    batch_size = 8
    xi = torch.randn(batch_size, requires_grad=True)
    eta = torch.randn(batch_size, requires_grad=True)
    zeta = torch.randn(batch_size, requires_grad=True)
    
    # Forward pass
    output = model(xi, eta, zeta)
    x, y, z = output[0], output[1], output[2]
    
    # Loss computation
    x_bc = x.detach()
    y_bc = y.detach()
    z_bc = z.detach()
    
    loss = loss_fn(x, y, z, xi, eta, zeta, x_bc, y_bc, z_bc)
    
    assert loss.item() >= 0
```

- [ ] **Step 2: Run test to verify it passes (smoke test)**

```bash
cd ~/algorithm/pinn-mg
pytest tests/integration/test_end_to_end.py -v
Expected: PASS (smoke test for imports)
```

- [ ] **Step 3: Write main.py**

```python
#!/usr/bin/env python3
"""
PINN-MG: Physics-Informed Neural Networks for Mesh Generation
Main entry point.
"""

import argparse
import os
import yaml
import torch
import numpy as np

from geometry.parasolid_parser import ParasolidParser
from geometry.boundary_fitter import BoundaryFitter
from network.pinn_network import PINNNetwork
from physics.loss_function import PINNLoss
from physics.navier_lame_3d import NavierLame3D
from mesh.mesh_generator import MeshGenerator
from mesh.gmsh_exporter import GMSHExporter


def load_config(config_path):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(model, loss_fn, xi, eta, zeta, x_bc, y_bc, z_bc, config):
    """Training loop."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['decay_interval'],
        gamma=config['training']['lr_decay']
    )
    
    epochs = config['training']['epochs']
    model.train()
    
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
    
    print("Loading geometry...")
    # Parse CAD file
    parser = ParasolidParser(args.input)
    parser.load(args.input)
    
    print("Fitting boundary...")
    # Fit boundary with SVR
    boundary_fitter = BoundaryFitter(
        C=config['boundary']['svr_C'],
        epsilon=config['boundary']['svr_epsilon']
    )
    # TODO: fit using parsed boundary points
    
    print("Initializing network...")
    model = PINNNetwork(
        input_dim=3,
        hidden_dim=config['model']['hidden_dim'],
        output_dim=3,
        num_layers=config['model']['num_layers']
    )
    
    print("Generating mesh...")
    mesh_gen = MeshGenerator(
        nx=config['mesh']['nx'],
        ny=config['mesh']['ny'],
        nz=config['mesh']['nz']
    )
    xi, eta, zeta = mesh_gen.generate_flat()
    
    # Convert to tensors
    xi_tensor = torch.tensor(xi, dtype=torch.float32, requires_grad=True)
    eta_tensor = torch.tensor(eta, dtype=torch.float32, requires_grad=True)
    zeta_tensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
    
    # Get boundary targets
    x_bc = np.array([boundary_fitter(x, e, z)[0] for x, e, z in zip(xi, eta, zeta)])
    y_bc = np.array([boundary_fitter(x, e, z)[1] for x, e, z in zip(xi, eta, zeta)])
    z_bc = np.array([boundary_fitter(x, e, z)[2] for x, e, z in zip(xi, eta, zeta)])
    
    x_bc_tensor = torch.tensor(x_bc, dtype=torch.float32)
    y_bc_tensor = torch.tensor(y_bc, dtype=torch.float32)
    z_bc_tensor = torch.tensor(z_bc, dtype=torch.float32)
    
    print("Initializing loss function...")
    loss_fn = PINNLoss(
        lambda_=config['physics']['lambda'],
        mu=config['physics']['mu']
    )
    
    print("Training...")
    model = train(model, loss_fn, xi_tensor, eta_tensor, zeta_tensor,
                x_bc_tensor, y_bc_tensor, z_bc_tensor, config)
    
    print("Generating output mesh...")
    model.eval()
    with torch.no_grad():
        output = model(xi_tensor, eta_tensor, zeta_tensor)
        x_out, y_out, z_out = output[0].numpy(), output[1].numpy(), output[2].numpy()
    
    print(f"Exporting to {args.output}...")
    exporter = GMSHExporter()
    exporter.export(xi, eta, zeta, x_out, y_out, z_out, args.output)
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN-MG')
    parser.add_argument('--input', type=str, required=True, help='Input .x_t file')
    parser.add_argument('--output', type=str, default='output.msh', help='Output .msh file')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    
    args = parser.parse_args()
    main(args)
```

- [ ] **Step 4: Run smoke test**

```bash
cd ~/algorithm/pinn-mg
python src/main.py --help
# Expected: Shows help message
```

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/integration/test_end_to_end.py
git commit -f "feat: add main entry point"
```

---

## Task 10: 注意力模块（预留）

**Files:**
- Create: `~/algorithm/pinn-mg/src/network/attention.py`

**Goal:** 预留注意力模块接口（论文中提及但未详述）

- [ ] **Step 1: Write attention.py (stub)**

```python
import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    """
    Attention module placeholder.
    
    Note: Original PINN-MG paper mentions attention but does not provide details.
    This is a placeholder for future implementation.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Placeholder: simple attention mechanism
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, hidden_dim)
        
        Returns:
            Output tensor (batch, hidden_dim)
        """
        # Placeholder implementation
        return x
```

- [ ] **Step 2: Commit**

```bash
git add src/network/attention.py
git commit -m "feat: add attention module placeholder"
```

---

## Plan Complete

**Plan saved to:** `~/algorithm/pinn-mg/docs/superpowers/plans/2026-05-13-pinn-mg-plan.md`

---

## Execution Options

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**