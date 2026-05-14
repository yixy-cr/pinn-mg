# PINN-MG Implementation Plan

## Project Overview
- **Name:** PINN-MG
- **Goal:** 从 CAD (.x_t Parasolid/MSH) 读取边界，通过 3D Navier-Lamé 方程约束训练 PINN 神经网络，生成六面体网格文件
- **Location:** ~/algorithm/pinn-mg/

## Status: IN_PROGRESS (修复调试中)

---

## 最近修复

### 问题1: 网格形状不对
- **根因:** 边界拟合使用随机输入，没有建立参数映射
- **修复:** 使用规格化参数域 → 物理域线性映射

### 问题2: GMSH文件无法读取
- **根因:** 自定义GMSH格式不正确
- **修复:** 使用meshio库正确导出

---

## Completed Tasks (2026-05-13)

| Task | Status | Description |
|------|--------|-------------|
| Task 0 | ✅ DONE | 项目初始化 |
| Task 1 | ✅ DONE | Parasolid解析器/边界加载 |
| Task 2 | ✅ DONE | 边界映射修复 |
| Task 3 | ✅ DONE | 输入增强 (tan/cot) |
| Task 4 | ✅ DONE | PINN网络架构 |
| Task 5 | ✅ DONE | 3D Navier-Lamé方程 |
| Task 6 | ✅ DONE | 损失函数 |
| Task 7 | ✅ DONE | 网格生成器 |
| Task 8 | ✅ DONE | GMSH导出修复 |
| Task 9 | ✅ DONE | 主程序 |
| Task 10 | ✅ DONE | 注意力模块(预留) |

---

## File Structure

```
pinn-mg/
├── src/
│   ├── geometry/
│   │   ├── parasolid_parser.py   # 支持加载.x_t和.msh边界
│   │   └── boundary_fitter.py    # SVR边界拟合
│   ├── network/
│   │   ├── pinn_network.py
│   │   ├── input_enhancement.py
│   │   └── attention.py
│   ├── physics/
│   │   ├── navier_lame_3d.py
│   │   └── loss_function.py
│   ├── mesh/
│   │   ├── mesh_generator.py
│   │   └── gmsh_exporter.py
│   └── main.py
├── 1pinflue10mm_g.msh           # 燃料棒边界mesh
├── configs/default.yaml
└── requirements.txt
```

---

## Usage

```bash
cd ~/algorithm/pinn-mg
PYTHONPATH=src python3 src/main.py --input 1pinflue10mm_g.msh --output output.msh --train
```

---

## Known Issues

1. 训练轮数较少(100轮)，网格可能需要更多训练
2. 边界映射是简单的线性映射，可改进为非线性