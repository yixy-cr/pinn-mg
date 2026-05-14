# PINN-MG: Physics-Informed Neural Networks with Multigrid Solver

## Date: 2026-05-13

---

## 1. Project Overview

**Project Name:** PINN-MG
**Goal:** 从 CAD 几何文件 (.x_t Parasolid 格式) 读取边界曲线，通过 PINN 神经网络求解 3D Navier-Lamé 方程，生成六面体（结构化）GMSH v4 网格文件。

**输入：**
- Parasolid .x_t 几何文件（包含边界曲线/曲面信息）
- 物理标签（壁面/入口/出口）

**输出：**
- GMSH v4 ASCII 格式六面体网格文件

---

## 2. Technical Specification

### 2.1 网络输出

**直接预测物理坐标，而非位移：**
- 输出：(x, y, z)
- 网络学习映射：(ξ, η, ζ) → (x, y, z)

### 2.2 方程：3D Navier-Lamé（在计算域上表达）

定义位移场 u = (x - ξ, y - η, z - ζ)，代入 Navier-Lamé 方程：

```
(λ+μ)∇(∇·u) + μ∇²u = 0
```

**Lamé 常数：**
- λ = 1.0
- μ = 0.35

**3 个分量方程（无独立连续性方程）：**
- f_x = (λ+μ)∂(∇·u)/∂x + μ∇²u_x = 0（在计算域 (ξ,η,ζ) 中表达）
- f_y = ...
- f_z = ...

⚠️ **关键**：方程必须在计算域上用 ∂x/∂ξ, ∂²x/∂ξ² 等偏导数表达，不能直接在物理域上计算。

### 2.3 网络架构

**输入增强（逐维）：**
```
U_in = [ξ, η, ζ, tan(ξ), tan(η), tan(ζ), cot(ξ), cot(η), cot(ζ)]
```

**前向传播：**
```
U_{i+1} = A_{i+1} · tanh(W_{i+1} U_i + b_{i+1})
```

**输出：** 3D 物理坐标 (x, y, z)

**注意力模块：** 预留接口，未来实现

### 2.4 坐标变换

**参数域 → 物理域（六面体结构化网格）：**
```
x = f_1(ξ, η, ζ)
y = f_2(ξ, η, ζ)
z = f_3(ξ, η, ζ)
```

网络直接学习这个映射。

### 2.5 损失函数

```
L_total = L_equation + L_boundary
```

**方程约束（在计算域上）：**
```
L_equation = (1/N_1) * Σ w_j * (||f_x||² + ||f_y||² + ||f_z||²)
```
其中 f_x, f_y, f_z 是变换后的 3D Navier-Lamé 方程（用偏导数表达）。

**边界约束：**
```
L_boundary = (1/N_2) * Σ w_k * (||x - f_1||² + ||y - f_2||² + ||z - f_3||²)
```
其中 f_1, f_2, f_3 是从 CAD 边界曲线拟合得到的 SVR 函数。

### 2.6 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 迭代次数 | 15,000 轮 |
| 学习率 | 初值 1e-5，每 1000 轮衰减 0.99 |
| 设备 | GPU |

---

## 3. Implementation Details（关键）

### 3.1 自动微分计算偏导数

在 navier_lame_3d.py 中实现：

```python
# x, y, z 是网络输出，xi, eta, zeta 是输入
# 计算一阶偏导
x_xi = torch.autograd.grad(x, xi, grad_outputs=torch.ones_like(x), create_graph=True)[0]
# 计算二阶偏导
x_xixi = torch.autograd.grad(x_xi, xi, grad_outputs=torch.ones_like(x_xi), create_graph=True)[0]
# 然后代入变换后的方程
```

### 3.2 边界条件来源

1. 从 CAD (.x_t) 提取边界曲线/曲面上的离散点
2. 用 SVR 拟合 x = f_1(ξ,η,ζ), y = f_2(ξ,η,ζ), z = f_3(ξ,η,ζ)
3. 对于六面体，边界是 6 个面，每个面可以用两个参数表示

---

## 4. Modules

### 4.1 几何处理模块

- `parasolid_parser.py`: 解析 .x_t 文件，提取边界
- `boundary_fitter.py`: SVR 拟合边界曲线（输出 f_1, f_2, f_3）

### 4.2 PINN 网络模块

- `pinn_network.py`: 神经网络架构
- `input_enhancement.py`: 输入增强（逐维 tan/cot）
- `attention.py`: 注意力模块（预留）

### 4.3 损失与训练模块

- `navier_lame_3d.py`: 3D Navier-Lamé 方程（在计算域上，自动微分）
- `loss_function.py`: 损失函数
- `trainer.py`: 训练循环

### 4.4 网格生成模块

- `mesh_generator.py`: 参数域六面体网格生成
- `gmsh_exporter.py`: GMSH v4 输出（六面体）

### 4.5 主程序

- `main.py`: 入口

---

## 5. Dependencies

```
torch >= 2.0
numpy
scikit-learn
gmsh (python binding)
```

---

## 6. File Structure

```
pinn-mg/
├── docs/
│   └── specs/
│       └── 2026-05-13-pinn-mg-design.md
├── src/
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── parasolid_parser.py
│   │   └── boundary_fitter.py
│   ├── network/
│   │   ├── __init__.py
│   │   ├── pinn_network.py
│   │   ├── input_enhancement.py
│   │   └── attention.py
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── navier_lame_3d.py
│   │   └── loss_function.py
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── mesh_generator.py
│   │   └── gmsh_exporter.py
│   └── main.py
├── configs/
│   └── default.yaml
├── requirements.txt
└── README.md
```

---

## 7. Implementation Order

1. Parasolid 解析器（读取 .x_t）
2. 边界 SVR 拟合（得到 f_1, f_2, f_3）
3. PINN 网络架构
4. **4a:** 推导变换后的 3D Navier-Lamé 方程（用偏导数表达）
   **4b:** 实现自动微分计算偏导数
   **4c:** 组装残差
5. 损失函数
6. 训练循环
7. 六面体网格生成
8. GMSH v4 输出

---

## 8. Notes

- 注意力机制为预留接口，论文中未详述
- 当前为单机 GPU，后续可扩展至集群 MPI+GPU
- Multigrid + IDW 插值暂不实现（论文中无）
- 网格类型：六面体（结构化），保持参数域映射 (ξ,η,ζ) → (x,y,z)
- 网络输出直接是物理坐标 (x,y,z)，不是位移场