# DeepCSI

DeepCSI 是一种基于对比源神经重参数化的端到端可微二维电磁逆散射方法, 其摆脱了大规模离线训练的依赖, 实现了无需训练的逐案例反演, 其反演过程与优化过程高度统一。

## 目录结构

```
DeepCSI/
├── data/                 # 示例数据与 Fresnel 实验数据（.exp）
├── src/              # 核心 Python 包
│   ├── __init__.py
│   ├── main.py           # CLI 入口（原 run_scripts_multifreq.sh 对应）
│   ├── generate_measurement.py  # 生成合成散射场
│   ├── models.py         # NeJF 网络与位置编码
│   ├── trainer.py        # InverseScatteringTrainer 主体
│   └── utils.py          # TV/MRTV 正则、Fresnel 数据读取、绘图函数等
├── run_multifreq.sh      # 使用多频合成数据的快速脚本
├── run_fresnel.sh        # Fresnel 实验数据示例脚本
└── results_*/            # 运行脚本后输出目录（自动生成）
```

## 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 1.12（推荐使用 GPU，如无 GPU 将自动退回 CPU 浮点）
- NumPy, Matplotlib, tqdm
- 可选：`scipy`（用于 Fresnel 数据的 `.mat` 校准文件）

可以通过以下命令快速创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib tqdm scipy
```

## 基础用法

### 生成合成测量 + 多频反演

```bash
cd DeepCSI
bash run_multifreq.sh
```

脚本会读取 `data/testcases` 下的 `.npy` 作为真实介电常数，先调用
`python -m src.generate_measurement` 生成前向散射，再运行
`python -m src.main` 完成 `fd-isp / pdtot-isp` 反演。结果会保存到
`results_multifreq/` 下的对应目录（同时输出 `.npy`、`loss_history.npy` 以及可视化 PNG），并记录指标至 `results_multifreq/result.csv`。

### Fresnel 实验数据示例

确保 `data/Fresnel` 目录下包含 `.exp` 与同名 `.npy` 参考文件，然后运行：

```bash
cd DeepCSI
bash run_fresnel.sh
```

脚本会针对指定实验 (`FoamDielExtTM` 等) 调用 `main.py` 完成反演，并记录指标至
`results_fresnel/result.csv`。

### 自定义 CLI

可以直接使用 Python 模块运行，传入任意参数组合：

```bash
python -m src.main \
  --expname demo \
  --basedir ./results_demo \
  --params_path ./data/testcases/epsilon_austria.npy \
  --recdata_path ./data/testcases/multifreq_measurement.npy \
  --method fd-isp \
  --freq 3,4,5, \
  --grid_num 64 \
  --N_inc 16 \
  --J_network multi-mlp
```

如果需要生成新的测量数据，可调用：

```bash
python -m src.generate_measurement \
  --params_path ./data/testcases/epsilon_austria.npy \
  --output ./data/testcases/measurement.npy \
  --freq 3,4,5,
```

## 结果输出说明

运行训练脚本时，会在 `results_*` 目录内生成：

- `loss_history.npy` / `loss_history.png`：记录各项损失曲线。
- `testset_XXXXXX.npy`：定期导出的网络预测电流/散射场/介电常数。
- `testset_XXXXXX.png` 与 `testset_XXXXXX_params.png`：电流幅度及重建 `epsilon` 的可视化。
- （可选）`result.csv`：当 `--result_file` 指定时，训练结束会写入一行指标条目。

## 文献引用
```bibtex
@article{sun2025physics,
  title={Physics-Informed Deep Contrast Source Inversion: A Unified Framework for Inverse Scattering Problems},
  author={Sun, Haoran and Liu, Daoqi and Zhou, Hongyu and Li, Maokun and Xu, Shenheng and Yang, Fan},
  journal={arXiv preprint arXiv:2508.10555},
  year={2025}
}
```

## E-mail
- maokunli@tsinghua.edu.cn
- sunhr23@mails.tsinghua.edu.cn