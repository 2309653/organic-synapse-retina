# -*- coding: utf-8 -*-
"""
"人工视网膜"仿真：感知-存储-识别一体（不训练神经网络）
======================================================

复现思路来自 Nature Photonics 2023（Chen et al., 有机光电突触做人工视网膜、
无需复杂 ANN 的人脸识别）的应用路线：

    图像亮度 → 光脉冲个数（亮的像素光照久 → 脉冲多）
            → 每个像素单元的电导变化（器件 = 探测器 + 存储器）
            → 电导图与"注册模板"做相似度比对 → 直接识别

与"训练神经网络"路线的本质区别：没有任何梯度训练，识别全靠
"感知时器件电导已经存下了图像信息 + 模板匹配"。代价是精度低于
训练型网络，换来的是极低的功耗和延迟。

器件非理想因素：
  --read-noise     读出噪声（每次读图加高斯噪声，随图像变化）
  --decay          读出前电导弛豫（对比度整体缩水）
  --cell-variation 像素间器件一致性（每个单元的电导窗口有固定偏差，
                   这是感知-存储应用的头号指标，对应文献中
                   "high-uniformity array" 强调的东西）

运行（在项目根目录下）::

    python src/retina.py                       # 默认 MNIST，单次运行
    python src/retina.py --sweep               # 扫描分辨率/衰减/一致性
    python src/retina.py --dataset fashion     # 换 Fashion-MNIST（见 README 下载说明）

输出：终端打印识别精度，figures/ 下生成图，results/ 存 CSV。
"""

import argparse
import csv
import math
import os
import sys

# Windows 上 torch 与 matplotlib 各自带了一份 OpenMP 运行库，需放宽重复检测
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torchvision
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------------
# 器件模型：与第一个仓库 (organic-synapse-mnist) 相同的指数电导更新模型
#   G(n+1) = G(n) + α·exp[−β·(G−G_min)/(G_max−G_min)]
# 这里预先算好"归一化电导表" x[n] ∈ [0,1]，像素单元 i 打 n 个脉冲后
#   G_i(n) = g_min + R_i · x[n]，R_i 为该单元自己的电导窗口（一致性偏差）。
# ----------------------------------------------------------------------------
def normalized_table(beta, n_pulses):
    """归一化电导表 x[n]：x[0]=0，n_pulses 个脉冲恰好扫满到 1。"""
    # α 由二分法标定
    lo, hi = 1e-12, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        x = [0.0]
        for _ in range(n_pulses):
            x.append(min(x[-1] + mid * math.exp(-beta * x[-1]), 1.0))
        if x[-1] < 1.0 - 1e-12:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    x = [0.0]
    for _ in range(n_pulses):
        x.append(min(x[-1] + alpha * math.exp(-beta * x[-1]), 1.0))
    return torch.tensor(x, dtype=torch.float32)


def cell_windows(R, n_pixels, variation, generator):
    """每个像素单元自己的电导窗口 R_i = R·(1 + v·N(0,1))（固定偏差，不随图变）。"""
    if variation <= 0:
        return torch.full((n_pixels,), R)
    z = torch.randn(n_pixels, generator=generator)
    return R * (1.0 + variation * z).clamp(min=0.1)


# ----------------------------------------------------------------------------
# 视网膜流程：编码（亮度→脉冲）→ 存储（脉冲→电导）→ 识别（模板匹配）
# ----------------------------------------------------------------------------
def encode_and_store(images, x_table, n_max, R_cells, decay, read_noise,
                     g_min, g_mean_R):
    """一批图像 (N, 1, 28, 28)，值域 [0,1] → 一批存储后的电导图 (N, 784)。"""
    pulses = torch.round(images.reshape(images.shape[0], -1) * n_max).long()
    G = g_min + R_cells * x_table[pulses]          # 感知-存储（查器件表）
    if decay > 0:                                  # 读出前的弛豫/衰减
        G = g_min + (G - g_min) * (1.0 - decay)
    if read_noise > 0:                             # 读出噪声（随图像随机）
        G = G + read_noise * g_mean_R * torch.randn_like(G)
    return G


def recognize(G_maps, templates):
    """最近模板识别：逐图与各类模板算余弦相似度（去均值后），取最大者。"""
    X = G_maps - G_maps.mean(dim=1, keepdim=True)
    T = templates - templates.mean(dim=1, keepdim=True)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    T = T / (T.norm(dim=1, keepdim=True) + 1e-9)
    return (X @ T.T).argmax(dim=1)


def build_templates(G_maps, labels, n_classes):
    """注册模板：每类取已存储电导图的平均（硬件上对应"登记已知样本"）。"""
    T = torch.zeros(n_classes, G_maps.shape[1])
    for c in range(n_classes):
        T[c] = G_maps[labels == c].mean(dim=0)
    return T


def load_data(name, data_dir, calib_per_class, n_test):
    tf = transforms.Compose([transforms.ToTensor()])   # 原始亮度 [0,1]
    ds = (torchvision.datasets.MNIST if name == "mnist"
          else torchvision.datasets.FashionMNIST)
    train = ds(data_dir, train=True, download=True, transform=tf)
    test = ds(data_dir, train=False, download=True, transform=tf)
    # 每类取 calib_per_class 张做"注册模板"用的样本
    idx, cnt = [], torch.zeros(10, dtype=torch.int32)
    for i in range(len(train)):
        y = train.targets[i]
        if cnt[y] < calib_per_class:
            idx.append(i)
            cnt[y] += 1
        if cnt.min() >= calib_per_class:
            break
    calib_x = torch.stack([train[i][0] for i in idx])
    calib_y = torch.tensor([int(train.targets[i]) for i in idx])
    test_x = torch.stack([test[i][0] for i in range(n_test)])
    test_y = torch.tensor([int(test.targets[i]) for i in range(n_test)])
    return calib_x, calib_y, test_x, test_y


def run_once(args, n_max=None, decay=None, cell_var=None):
    n_max = args.n_max if n_max is None else n_max
    decay = args.decay if decay is None else decay
    cell_var = args.cell_variation if cell_var is None else cell_var
    calib_x, calib_y, test_x, test_y = load_data(
        args.dataset, args.data_dir, args.calib, args.test)

    x_table = normalized_table(args.beta_p, n_max)
    gen = torch.Generator().manual_seed(args.seed)   # 固定的器件阵列
    R_cells = cell_windows(args.gmax - args.gmin, 784, cell_var, gen)

    # 模板 = 刚感知完就注册（新鲜、未衰减）；测试图像在等待比对期间
    # 会经历衰减（decay 只作用在测试图上）——这是"存着存着就变淡"的
    # 真实场景，也是保持特性(retention)影响识别的机制。
    cal_G = encode_and_store(calib_x, x_table, n_max, R_cells, 0.0,
                             args.read_noise, args.gmin, args.gmax - args.gmin)
    templates = build_templates(cal_G, calib_y, 10)
    test_G = encode_and_store(test_x, x_table, n_max, R_cells, decay,
                              args.read_noise, args.gmin, args.gmax - args.gmin)
    pred = recognize(test_G, templates)
    return ((pred == test_y).float().mean().item(),
            (test_x, test_G, calib_x, cal_G, calib_y))


def main():
    ap = argparse.ArgumentParser(description="人工视网膜仿真（感知-存储-识别一体）")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    ap.add_argument("--calib", type=int, default=100, help="每类注册模板用的样本数")
    ap.add_argument("--test", type=int, default=2000, help="测试图像数")
    ap.add_argument("--n-max", type=int, default=30, help="全亮像素的光脉冲数上限（分辨率）")
    ap.add_argument("--beta-p", type=float, default=2.0, help="器件非线性因子 β")
    ap.add_argument("--gmin", type=float, default=0.1)
    ap.add_argument("--gmax", type=float, default=9.9)
    ap.add_argument("--decay", type=float, default=0.0, help="读出前电导衰减比例 [0,1)")
    ap.add_argument("--read-noise", type=float, default=0.15, help="读噪声 σ（窗口比例）")
    ap.add_argument("--cell-variation", type=float, default=0.0,
                    help="像素间电导窗口相对偏差 σ（如 0.1 = ±10%%）")
    ap.add_argument("--sweep", action="store_true", help="扫描分辨率/衰减/一致性的影响")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    acc, (test_x, test_G, calib_x, cal_G, calib_y) = run_once(args)
    print(f"数据集: {args.dataset} | 分辨率 n_max={args.n_max} | 衰减={args.decay} | "
          f"读噪声={args.read_noise} | 像素间偏差={args.cell_variation}")
    print(f"人工视网膜模板匹配识别精度: {acc*100:.2f}%   "
          f"(对照: 训练型 MLP ~96%, 见 organic-synapse-mnist 仓库)")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # ---- 图1：流程可视化（输入亮度 vs 存储后的电导图）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    for c in range(5):
        i = int((calib_y == c).nonzero()[0][0])     # 每类第一个注册样本
        axes[0, c].imshow(calib_x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"class {c}: input")
        axes[1, c].imshow(cal_G[i].reshape(28, 28), cmap="inferno")
        axes[1, c].set_title("stored conductance")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Artificial retina: brightness → light pulses → stored conductance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_retina_pipeline.png"), dpi=150)
    plt.close(fig)

    if not args.sweep:
        print("图已保存: figures/fig1_retina_pipeline.png")
        return

    # ---- 扫描：分辨率 / 衰减 / 像素间一致性 ----
    rows = []
    for nm in [5, 10, 20, 30, 60, 100]:
        a, _ = run_once(args, n_max=nm)
        rows.append(("resolution", nm, args.decay, args.cell_variation, a))
        print(f"  n_max={nm:3d}            → acc={a*100:5.2f}%")
    for d in [0.0, 0.2, 0.4, 0.6]:
        a, _ = run_once(args, decay=d)
        rows.append(("decay", args.n_max, d, args.cell_variation, a))
        print(f"  decay={d:.1f}           → acc={a*100:5.2f}%")
    for v in [0.0, 0.05, 0.1, 0.2, 0.3]:
        a, _ = run_once(args, cell_var=v)
        rows.append(("cell_var", args.n_max, args.decay, v, a))
        print(f"  cell_variation={v:.2f}  → acc={a*100:5.2f}%")

    with open(os.path.join(RES_DIR, "retina_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "n_max", "decay", "cell_variation", "accuracy"])
        w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("resolution", "max pulses per pixel $n_{max}$", "Resolution"),
              ("decay", "conductance decay before readout", "Retention"),
              ("cell_var", "cell-to-cell variation $\\sigma$", "Uniformity")]
    for ax, (key, xlabel, title) in zip(axes, panels):
        pts = [(r[1 if key == "resolution" else 2 if key == "decay" else 3], r[4])
               for r in rows if r[0] == key]
        ax.plot(*zip(*pts), "o-")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_title(title + " vs. accuracy")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_retina_sweep.png"), dpi=150)
    print("图已保存: figures/fig1_retina_pipeline.png, figures/fig2_retina_sweep.png")


if __name__ == "__main__":
    main()
