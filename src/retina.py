# -*- coding: utf-8 -*-
"""
"人工视网膜"仿真 —— 严格按论文补充材料（SI）的真实方法实现
==========================================================

方法出处：Chen et al., Nature Photonics 17, 629 (2023)，补充材料
"Simulation of a single layer of synapse array as artificial retina for face
recognition"（SI p.20-22）。论文方法为**期望阈值法**（expectation-based
thresholding）的单目标人脸识别，共五个环节：

  ① 器件"写"模型：光强 → 非易失电流增量，二次函数拟合（SI Fig. S28a；
     实测：空白器件打 60 s 不同光强脉冲，记录每次的非易失电流）
  ② 器件"读"模型：(光强, 记忆电流) → 瞬态光电流，函数曲面拟合
     （SI Fig. S28b；实测：2 s 光脉冲重复刺激，逐脉冲记录记忆电流与
     电流增量，对(光强, 记忆)拟合曲面）
  ③ 训练（写记忆）：目标人物的 9 张人脸图，灰度归一化到器件探测动态
     范围（原文 2-7 mW/cm²）后逐张写入 64×64 阵列 → 每像素记忆电流
     （SI Fig. 6a）
  ④ 特征器件+阈值：记忆电流前 25%（75% 分位）像素 = 特征器件（自动
     圈出目标脸的关键区域）；用读模型预测"正样本到来时"各特征器件的
     瞬态光电流（对 9 张训练图取平均），乘容差比 0.9 → 每器件判决
     阈值（SI Fig. S28c）
  ⑤ 测试（判决）：特征器件的实际光电流 ≥ 阈值 → active（红），否则
     idle（蓝）；激活率 = active 占比。原文结果：正样本 88-91%，
     负样本 <70%（SI Fig. 6c,d）

两种任务模式：
  --task orl   （默认，忠实复现）ORL 人脸库：40 人×10 张、64×64——与论文
               同款数据集与协议。选一个目标人，9 张训练、第 10 张做正
               样本（另加高斯噪声变体），其余 39 人 390 张做负样本，
               按论文以 70% 激活率为二分界。
  --task mnist （对照演示）MNIST 十分类：每类一个视网膜，取激活率最高的
               类。演示该方法的适用边界：它为"刚性目标验证"设计，对
               笔迹变化剧烈的数字类效果有限。

注意：视网膜路线的器件模型是"两个拟合函数"（①②），**不是** α·exp(−βx)
指数模型——那是网络训练路线（差分对 + HDBP）用的，见姊妹仓库
organic-synapse-mnist。本脚本的 write_model 与 response_model 为**占位
模型**（凸二次写入 + 记忆平方增益读出），接入实测器件时用你自己按
SI Fig. S28 方法定义的拟合结果替换这两个函数即可；替换后激活率数字
会随之改变（见 README 的讨论）。

ORL 数据获取（约 1.2 MB）::

    git clone --depth 1 https://github.com/lloydmeta/Olivetti-PNG.git data/ORL

运行::

    python src/retina.py                       # ORL 人脸复现 + 流程图
    python src/retina.py --sweep               # 非理想因素扫描
    python src/retina.py --task mnist          # MNIST 对照
"""

import argparse
import csv
import os

# Windows 上 torch 与 matplotlib 各自带了一份 OpenMP 运行库，需放宽重复检测
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
ORL_DIR = os.path.join(ROOT, "data", "ORL")


# ----------------------------------------------------------------------------
# ① 器件"写"模型：光强 I → 非易失电流增量（占位凸二次函数，用实测拟合替换）
#    Δm = I²：记忆集中在最亮像素（人脸高光区），随像素亮度超线性增长；
#    凸形比凹形区分度更好（消融实验见 README）
# ----------------------------------------------------------------------------
def write_model(I):
    return I * I


# ----------------------------------------------------------------------------
# ② 器件"读"模型：(光强 I, 记忆 m) → 瞬态光电流（占位曲面，用实测拟合替换）
#    物理直觉：(a) 存在探测阈值——低于 ~0.3 的光强几乎不产生瞬态响应
#    （压制暗背景与读出噪声）；(b) 响应随记忆**平方**增强（记忆越深 →
#    沟道掺杂越多 → 光导增益越大，增益项 0.1+0.9m²）
# ----------------------------------------------------------------------------
def response_model(I, m):
    return torch.relu(I - 0.3) * (0.1 + 0.9 * m * m) * 2.5


def quantize(I, n_levels):
    """光强量化到 n_levels 档（器件可分辨的强度级别数）。"""
    if n_levels <= 1:
        return I
    return torch.round(I * (n_levels - 1)) / (n_levels - 1)


def to_dynamic_range(pixels, i_lo, i_hi):
    """像素灰度 [0,1] → 器件探测动态范围（论文：2-7 mW/cm² 的归一化）。"""
    return i_lo + (i_hi - i_lo) * pixels


# ----------------------------------------------------------------------------
# ③④ 训练一个视网膜：写入记忆 → 选特征器件 → 定判决阈值
# ----------------------------------------------------------------------------
def train_retina(X, args):
    """X: (N, P) 已归一化到动态范围的光强图。返回记忆/特征/阈值。"""
    X = quantize(X, args.n_levels)
    m = write_model(X).mean(dim=0)                          # ③ 平均记忆电流
    q = torch.quantile(m, args.feature_quantile)
    feat = m >= q                                           # ④ 特征器件(前25%)
    exp_r = response_model(X, m.unsqueeze(0)).mean(dim=0)   # 预测正样本响应
    return {"memory": m, "feat": feat, "thr": exp_r[feat]}


# ----------------------------------------------------------------------------
# ⑤ 测试：预测光电流响应 → 逐器件判决 → 激活率
# ----------------------------------------------------------------------------
def test_activation(X, retina, args, gains=None):
    """返回 (激活率 (N,), 全尺寸 active 掩码 (N, P))。gains 为像素间光响应增益。"""
    X = quantize(X, args.n_levels)
    m = retina["memory"] * (1.0 - args.decay)     # 记忆衰减：保持特性变差
    rho = response_model(X, m.unsqueeze(0))       # 预测瞬态光电流
    if gains is not None:
        rho = rho * gains.unsqueeze(0)            # 像素间光响应差异(固定偏差)
    if args.read_noise > 0:
        rho = rho + args.read_noise * torch.randn_like(rho)
    active = rho[:, retina["feat"]] >= (args.tolerance * retina["thr"]).unsqueeze(0)
    full = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool)
    full[:, retina["feat"]] = active              # 映射回全像素网格（画图用）
    return active.float().mean(dim=1), full


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def load_orl(orl_dir=ORL_DIR):
    """ORL/Olivetti：400 张 64×64 灰度人脸，每 10 张为同一人（按人分组）。"""
    img_dir = os.path.join(orl_dir, "images")
    files = sorted(os.listdir(img_dir),
                   key=lambda f: int(f.split("-")[1].split(".")[0]))
    X = torch.stack([
        torch.tensor(np.array(Image.open(os.path.join(img_dir, f))
                              .convert("L"), dtype=np.float32) / 255.0)
        for f in files]).reshape(len(files), -1)
    y = torch.arange(len(files)) // 10
    return X, y


def load_mnist(data_dir, per_class, n_test):
    tf = transforms.Compose([transforms.ToTensor()])   # 原始灰度 [0,1]
    train = torchvision.datasets.MNIST(data_dir, train=True,
                                       download=True, transform=tf)
    test = torchvision.datasets.MNIST(data_dir, train=False,
                                      download=True, transform=tf)
    xs, ys = [], []
    for c in range(10):
        idx = (train.targets == c).nonzero()[:per_class].squeeze(1)
        xs.append(torch.stack([train[int(i)][0] for i in idx]))
        ys.append(torch.full((len(idx),), c))
    train_x = torch.cat(xs).reshape(-1, 784)
    train_y = torch.cat(ys)
    n_test = min(n_test, len(test))
    test_x = torch.stack([test[i][0] for i in range(n_test)]).reshape(-1, 784)
    test_y = test.targets[:n_test].clone()
    return train_x, train_y, test_x, test_y


def make_gains(n_pixels, args):
    """固定的像素间光响应增益 g_j = 1 + σ·N(0,1)（同一阵列内不变）。"""
    gen = torch.Generator().manual_seed(args.seed + 12345)
    return (1.0 + args.cell_variation
            * torch.randn(n_pixels, generator=gen)).clamp(min=0.1)


# ----------------------------------------------------------------------------
# ORL 任务：论文协议的忠实复现
# ----------------------------------------------------------------------------
def run_orl(args):
    if not os.path.isdir(ORL_DIR):
        raise SystemExit(
            "未找到 ORL 数据。请先执行:\n"
            "  git clone --depth 1 "
            "https://github.com/lloydmeta/Olivetti-PNG.git data/ORL")
    X, y = load_orl()
    X = to_dynamic_range(X, args.i_lo, args.i_hi)     # 归一化到器件动态范围
    n_px = X.shape[1]

    tgt = args.target
    tr = X[tgt * 10: tgt * 10 + 9]                    # 9 张训练（写记忆）
    pos = X[tgt * 10 + 9: tgt * 10 + 10]              # 第 10 张 = 正样本
    torch.manual_seed(args.seed)
    noisy = (pos + args.noise_sigma * torch.randn(args.n_noisy, *pos.shape[1:]))
    pos_all = torch.cat([pos, noisy.clamp(0, args.i_hi)])   # 正样本+噪声变体
    neg = X[y != tgt]                                 # 其余 39 人 = 负样本

    retinas = {"target": train_retina(tr, args)}
    gains = make_gains(n_px, args) if args.cell_variation > 0 else None
    r = retinas["target"]

    pos_rate, pos_act = test_activation(pos_all, r, args, gains)
    neg_rate, neg_act = test_activation(neg, r, args, gains)

    print(f"目标人物: #{tgt}（前 9 张训练，第 10 张+{args.n_noisy}个噪声变体为正样本，"
          f"其余 39 人 {len(neg)} 张为负样本）")
    print(f"正样本激活率: 干净={pos_rate[0]*100:.0f}%  "
          f"噪声均值={pos_rate[1:].mean()*100:.0f}%   （论文: 88-91%）")
    print(f"负样本激活率: mean={neg_rate.mean()*100:.0f}%  "
          f"max={neg_rate.max()*100:.0f}%            （论文: <70%）")
    tp = (pos_rate >= args.cutoff).float().mean().item()
    tn = (neg_rate < args.cutoff).float().mean().item()
    print(f"以 {args.cutoff:.0%} 为界二分: 正样本通过率={tp*100:.0f}%  "
          f"负样本拒绝率={tn*100:.0f}%")
    return {"X": X, "y": y, "retina": r, "gains": gains, "args": args,
            "pos_rate": pos_rate, "pos_act": pos_act, "neg_rate": neg_rate,
            "neg_act": neg_act, "tgt": tgt}


def orl_sweep(args):
    """四个非理想因素 × (正/负激活率 + 二分精度)。"""
    factors = [
        ("decay", [0.0, 0.2, 0.4, 0.6], "记忆衰减 (保持特性)"),
        ("cell_variation", [0.0, 0.05, 0.1, 0.2, 0.3], "像素间光响应差异"),
        ("read_noise", [0.0, 0.05, 0.1, 0.2], "读噪声 σ"),
        ("tolerance", [0.8, 0.85, 0.9, 0.95, 1.0], "判决容差比"),
    ]
    rows = []
    for key, values, label in factors:
        for v in values:
            a = argparse.Namespace(**vars(args))
            setattr(a, key, v)
            res = run_orl(a)
            pr, nr = res["pos_rate"], res["neg_rate"]
            acc = 0.5 * ((pr >= args.cutoff).float().mean().item()
                         + (nr < args.cutoff).float().mean().item())
            rows.append((key, v, pr[0].item(), pr[1:].mean().item(),
                         nr.mean().item(), nr.max().item(), acc))
            print(f"  {label} = {v:<4} → 正样本 {pr[0]*100:3.0f}%/{pr[1:].mean()*100:3.0f}%"
                  f"  负样本 mean {nr.mean()*100:3.0f}%  二分 acc {acc*100:3.0f}%\n")

    with open(os.path.join(RES_DIR, "orl_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "value", "pos_clean", "pos_noisy",
                    "neg_mean", "neg_max", "binary_acc"])
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (key, values, label) in zip(axes, factors):
        pts = [(r[1], r[2], r[4]) for r in rows if r[0] == key]
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], "o-", color="tab:red", label="positive")
        ax.plot(xs, [p[2] for p in pts], "s-", color="tab:blue", label="negative (mean)")
        ax.axhline(args.cutoff, color="gray", ls="--", lw=1, label="cutoff 70%")
        ax.set_xlabel(label)
        ax.set_ylabel("Activation rate")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Device non-idealities vs. retina activation separation (ORL)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_retina_sweep.png"), dpi=150)
    print("图已保存: figures/fig2_retina_sweep.png")


def orl_fig1(res):
    """论文 Fig.6 风格的流程图：训练图 → 记忆 → 特征 → 激活图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args, r, tgt = res["args"], res["retina"], res["tgt"]
    X = res["X"]
    H = W = 64

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 10, height_ratios=[1, 1.6])
    for k in range(9):                              # 行1: 9 张训练图
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(X[tgt * 10 + k].reshape(H, W), cmap="gray",
                  vmin=args.i_lo, vmax=args.i_hi)
        ax.set_title(f"train {k+1}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 9])
    ax.imshow(X[tgt * 10 + 9].reshape(H, W), cmap="gray",
              vmin=args.i_lo, vmax=args.i_hi)
    ax.set_title("test (pos)", fontsize=8, color="tab:red")
    ax.axis("off")

    panels = [  # 行2: 记忆 / 特征 / 正样本激活 / 噪声正样本 / 负样本激活
        ("③ memory", r["memory"].reshape(H, W), "inferno", None),
        ("④ feature devices", r["feat"].reshape(H, W).float(), "gray", None),
        ("⑤ pos active "
         f"{res['pos_rate'][0]*100:.0f}%", res["pos_act"][0], "act", (H, W, r)),
        ("⑤ pos+noise "
         f"{res['pos_rate'][1:].mean()*100:.0f}%", res["pos_act"][1], "act", (H, W, r)),
        ("⑤ neg example "
         f"{res['neg_rate'][torch.argmax(res['neg_rate'])]*100:.0f}%",
         res["neg_act"][int(torch.argmax(res['neg_rate']))], "act", (H, W, r)),
    ]
    for k, (title, data, cm, extra) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        if cm == "act":
            Hh, Ww, rr = extra
            feat2d = rr["feat"].reshape(Hh, Ww)
            rgb = torch.zeros(Hh, Ww, 3)
            rgb[feat2d & data.reshape(Hh, Ww)] = torch.tensor([1.0, 0.25, 0.25])
            rgb[feat2d & ~data.reshape(Hh, Ww)] = torch.tensor([0.3, 0.45, 1.0])
            ax.imshow(rgb)
        else:
            ax.imshow(data, cmap=cm)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Artificial retina, expectation-based thresholding "
                 "(per Nat. Photonics 2023 SI p.20-22)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_retina_pipeline.png"), dpi=150)
    plt.close(fig)
    print("图已保存: figures/fig1_retina_pipeline.png")


# ----------------------------------------------------------------------------
# MNIST 对照任务：方法的适用边界
# ----------------------------------------------------------------------------
def run_mnist(args):
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, args.calib, args.test)
    train_x = to_dynamic_range(train_x, args.i_lo, args.i_hi)
    test_x = to_dynamic_range(test_x, args.i_lo, args.i_hi)

    retinas = [train_retina(train_x[train_y == c], args) for c in range(10)]
    gains = make_gains(784, args) if args.cell_variation > 0 else None

    A = torch.zeros(len(test_x), 10)
    for c, r in enumerate(retinas):
        A[:, c], _ = test_activation(test_x, r, args, gains)
    pred = A.argmax(dim=1)
    acc = (pred == test_y).float().mean().item()

    idx = torch.arange(len(test_y))
    true_A = A[idx, test_y]
    A_other = A.clone()
    A_other[idx, test_y] = -1
    rival_A = A_other.max(dim=1).values

    print(f"MNIST 十分类（每类一个视网膜，激活率 argmax）: {acc*100:.2f}%")
    print(f"真类激活率 mean={true_A.mean():.3f} vs 最强竞争类 mean={rival_A.mean():.3f}")
    print("（对照: 训练型 MLP ~96%，见 organic-synapse-mnist；"
          "该方法面向刚性目标验证，数字笔迹变化大，分离度有限——见 README）")
    return acc


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="人工视网膜仿真（期望阈值法，按论文 SI 实现）")
    ap.add_argument("--task", choices=["orl", "mnist"], default="orl")
    # ORL 协议
    ap.add_argument("--target", type=int, default=7, help="目标人物编号(0-39)")
    ap.add_argument("--noise-sigma", type=float, default=0.1,
                    help="正样本高斯噪声强度")
    ap.add_argument("--n-noisy", type=int, default=5, help="噪声正样本个数")
    ap.add_argument("--cutoff", type=float, default=0.7, help="二分激活率界(论文0.7)")
    ap.add_argument("--i-lo", type=float, default=None,
                    help="动态范围下限(归一化，论文2mW/cm²→0.2)")
    ap.add_argument("--i-hi", type=float, default=None,
                    help="动态范围上限(归一化，论文7mW/cm²→0.7)")
    # MNIST
    ap.add_argument("--calib", type=int, default=200, help="每类训练图像数")
    ap.add_argument("--test", type=int, default=2000, help="测试图像数")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    # 共同参数
    ap.add_argument("--n-levels", type=int, default=32, help="光强量化档数")
    ap.add_argument("--feature-quantile", type=float, default=0.75,
                    help="特征器件记忆分位(0.75=前25%%，论文值)")
    ap.add_argument("--tolerance", type=float, default=0.85,
                    help="判决容差比(论文取0.9；本占位模型下0.85最优，见README)")
    ap.add_argument("--decay", type=float, default=0.0, help="训练后记忆衰减[0,1)")
    ap.add_argument("--read-noise", type=float, default=0.02, help="读噪声σ")
    ap.add_argument("--cell-variation", type=float, default=0.0,
                    help="像素间光响应相对偏差σ")
    ap.add_argument("--sweep", action="store_true", help="扫描非理想因素")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.i_lo is None:
        args.i_lo = 0.2 if args.task == "orl" else 0.0
    if args.i_hi is None:
        args.i_hi = 0.7 if args.task == "orl" else 1.0

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.task == "orl":
        if args.sweep:
            orl_sweep(args)
        else:
            res = run_orl(args)
            orl_fig1(res)
    else:
        run_mnist(args)


if __name__ == "__main__":
    main()
