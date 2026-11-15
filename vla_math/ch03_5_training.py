"""演習 3.5: 深いネットを学習させる工夫

実行: python3 vla_math/ch03_5_training.py
出力: docs/vla-math/figures/ch03_5_init.png, ch03_5_gradflow.png, ch03_5_deep_train.png, ch03_5_optim.png
必要: numpy, matplotlib（GPU は不要. 2 分ほど）

(1) 初期値の大きさ: 50 層のネットに乱数を通し、層ごとの値の大きさ（標準偏差）を測る
(2) 勾配の大きさ: 20 層のネットで、層ごとの重みの勾配の大きさを測る（シグモイド、tanh、ReLU、残差接続）
(3) 20 層と 100 層のネットに渦巻きを学習させる. 残差接続と正規化があると、深くても学習が進む
(4) 細長い谷（2 次関数）で、勾配降下法・モーメンタム・Adam の進み方を比べる
(5) ミニバッチの勾配のばらつきは、バッチの大きさ B の平方根に反比例して小さくなる
(6) バッチ正規化と層正規化は、平均と分散を取る向きが違う
"""

from __future__ import annotations

import numpy as np

from ch03_mlp import MLP, make_spiral, softmax
from common import C_A, C_B, C_C, C_INK, C_MUTED, saved, setup_matplotlib, FIG_DIR


def sig(z):
    return 1 / (1 + np.exp(-z))


ACTS = {   # 活性化関数と、その微分（出力 a と入力 z から計算）
    "sigmoid": (sig, lambda a, z: a * (1 - a)),
    "tanh": (np.tanh, lambda a, z: 1 - a * a),
    "relu": (lambda z: np.maximum(z, 0), lambda a, z: (z > 0).astype(float)),
}


def layernorm(x, eps=1e-5):
    mu = x.mean(1, keepdims=True)
    sd = np.sqrt(x.var(1, keepdims=True) + eps)
    return (x - mu) / sd, sd


def layernorm_back(dy, y, sd):
    """y = (x − 平均) / 標準偏差 の逆伝播（学習する拡大・平行移動は省く）."""
    return (dy - dy.mean(1, keepdims=True) - y * (dy * y).mean(1, keepdims=True)) / sd


class DeepNet:
    """2 → 幅 n → （n → n を depth 回）→ 2.

    普通のネット: h ← f(h W + b)
    残差接続:     h ← h + f(LN(h) W + b)  （LN は層正規化. 出力の前にも LN を 1 回）
    """

    def __init__(self, depth, width, act, residual, rng, gain):
        self.f, self.df = ACTS[act]
        self.residual = residual
        self.Win = rng.normal(0, 1 / np.sqrt(2), (2, width))
        self.bin = np.zeros(width)
        self.W = [rng.normal(0, gain / np.sqrt(width), (width, width)) for _ in range(depth)]
        self.b = [np.zeros(width) for _ in range(depth)]
        self.Wout = rng.normal(0, 1 / np.sqrt(width), (width, 2))
        self.bout = np.zeros(2)

    def params(self):
        return [self.Win, self.bin] + self.W + self.b + [self.Wout, self.bout]

    def loss_and_grads(self, x, y):
        h = x @ self.Win + self.bin
        cache = []
        for W, b in zip(self.W, self.b):
            u, sd = layernorm(h) if self.residual else (h, None)
            z = u @ W + b
            a = self.f(z)
            cache.append((u, sd, z, a))
            h = h + a if self.residual else a
        hf, sdf = layernorm(h) if self.residual else (h, None)
        p = softmax(hf @ self.Wout + self.bout)
        n = len(x)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
        g = p.copy()
        g[np.arange(n), y] -= 1
        g /= n
        dWout, dbout = hf.T @ g, g.sum(0)
        dh = g @ self.Wout.T
        if self.residual:
            dh = layernorm_back(dh, hf, sdf)
        dW, db = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            u, sd, z, a = cache[i]
            dz = dh * self.df(a, z)
            dW[i], db[i] = u.T @ dz, dz.sum(0)
            du = dz @ self.W[i].T
            dh = dh + layernorm_back(du, u, sd) if self.residual else du
        dWin, dbin = x.T @ dh, dh.sum(0)
        return loss, [dWin, dbin] + dW + db + [dWout, dbout], dW


def momentum_train(net, x, y, steps, lr, beta=0.9):
    vel = [np.zeros_like(p) for p in net.params()]
    hist = []
    for _ in range(steps):
        loss, grads, _ = net.loss_and_grads(x, y)
        hist.append(loss)
        for p, g, v in zip(net.params(), grads, vel):
            v *= beta
            v += g
            p -= lr * v
    return hist


# ---------------------------------------------------------------- (4) 細長い谷
A = np.array([1.0, 25.0])          # f(θ) = ½ (θ1² + 25 θ2²)


def quad(th):
    return 0.5 * np.sum(A * th ** 2)


def run_opt(kind, lr, steps, th0=(-4.0, 1.5), beta=0.9, b2=0.999):
    th = np.array(th0, float)
    m, v = np.zeros(2), np.zeros(2)
    path = [th.copy()]
    for t in range(1, steps + 1):
        g = A * th
        if kind == "gd":
            th = th - lr * g
        elif kind == "momentum":
            m = beta * m + g
            th = th - lr * m
        else:   # adam
            m = beta * m + (1 - beta) * g
            v = b2 * v + (1 - b2) * g * g
            th = th - lr * (m / (1 - beta ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        path.append(th.copy())
    return np.array(path)


def steps_to(path, tol):
    """f が tol 未満になり、そのあと最後まで tol 未満のままでいる最初の歩数（そうならなければ None）."""
    vals = np.array([quad(p) for p in path])
    above = np.flatnonzero(~(vals < tol))
    if len(above) == 0:
        return 0
    k = int(above[-1]) + 1
    return k if k < len(vals) else None


def main():
    plt = setup_matplotlib()
    rng = np.random.default_rng(0)

    print("(1) 50 層・幅 256 のネットに標準正規分布の入力 1000 個を通す. 重みは N(0, (c/√n)²)")
    x0 = rng.standard_normal((1000, 256))
    init_rows = {}
    for act, cs in (("tanh", (0.5, 1.0, 2.0)), ("relu", (1.0, np.sqrt(2), 2.0))):
        f = ACTS[act][0]
        for c in cs:
            r = np.random.default_rng(1)
            h, stds, Ws, zs = x0, [x0.std()], [], []
            for _ in range(50):
                Ws.append(r.normal(0, c / np.sqrt(256), (256, 256)))
                zs.append(h @ Ws[-1])
                h = f(zs[-1])
                stds.append(h.std())
            init_rows[(act, c)] = stds
            g = np.random.default_rng(2).standard_normal(h.shape)     # 出口から勾配を逆向きに 50 層戻す
            g_top = np.linalg.norm(g)
            for W, z in zip(reversed(Ws), reversed(zs)):
                g = (g * ACTS[act][1](f(z), z)) @ W.T
            extra = f"、|値| > 0.9 の割合 {np.mean(np.abs(h) > 0.9):.0%}" if act == "tanh" else ""
            print(f"    {act:4s} c = {c:.2f}: 標準偏差 1 層 {stds[1]:.3g}, 10 層 {stds[10]:.3g}, "
                  f"50 層 {stds[50]:.3g}{extra}. 勾配を 50 層戻すと {np.linalg.norm(g) / g_top:.2g} 倍")

    print("\n(2) 20 層・幅 64 のネットで、学習前の層ごとの重みの勾配の大きさ（渦巻き 400 点の交差エントロピー）")
    xs, ys = make_spiral(np.random.default_rng(0), 200)
    configs = [("シグモイド", "sigmoid", False, 1.0), ("tanh", "tanh", False, 1.0),
               ("ReLU（He）", "relu", False, np.sqrt(2)), ("ReLU + 残差 + LN", "relu", True, np.sqrt(2))]
    grad_rows = {}
    for label, act, res, gain in configs:
        net = DeepNet(20, 64, act, res, np.random.default_rng(2), gain)
        _, _, dW = net.loss_and_grads(xs, ys)
        norms = [np.linalg.norm(g) for g in dW]
        grad_rows[label] = norms
        print(f"    {label:16s}: 1 層目 {norms[0]:.2e}、10 層目 {norms[9]:.2e}、20 層目 {norms[19]:.2e}"
              f"（1 層目 / 20 層目 = {norms[0] / norms[19]:.1e}）")

    print("\n(3) 幅 32 のネットに渦巻きを学習させる（モーメンタム 0.9、全 400 点）")
    train_rows = {}
    for depth, steps, runs in ((20, 1000, (("シグモイド", "sigmoid", False, 1.0, 0.05),
                                           ("ReLU（He）", "relu", False, np.sqrt(2), 0.01),
                                           ("ReLU + 残差 + LN", "relu", True, np.sqrt(2), 0.01))),
                               (100, 500, (("ReLU（He）", "relu", False, np.sqrt(2), 0.003),
                                           ("ReLU（He）", "relu", False, np.sqrt(2), 0.01),
                                           ("ReLU + 残差 + LN", "relu", True, np.sqrt(2), 0.003),
                                           ("ReLU + 残差 + LN", "relu", True, np.sqrt(2), 0.01)))):
        print(f"    {depth} 層、{steps} 歩:")
        for label, act, res, gain, lr in runs:
            net = DeepNet(depth, 32, act, res, np.random.default_rng(3), gain)
            hist = momentum_train(net, xs, ys, steps + 1, lr)
            train_rows[(depth, label, lr)] = hist
            marks = "、".join(f"{hist[i]:.3f}（{i} 歩）" for i in (0, 100, 300, steps) if i <= steps)
            print(f"      {label:16s} 学習率 {lr:<5}: 損失 {marks}")

    print("\n(4) f(θ) = ½(θ1² + 25 θ2²) を (−4, 1.5) から小さくする. 学習率などは候補の中で最速のもの")
    print("    （速さ = f が 10⁻⁶ 未満になり、そのあと 3000 歩まで 10⁻⁶ 未満のままでいる最初の歩数）")
    best = {}
    for kind, grid, betas in (("gd", np.linspace(0.005, 0.079, 75), (0.0,)),
                              ("momentum", np.linspace(0.005, 0.2, 40), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
                              ("adam", np.geomspace(0.001, 1.0, 120), (0.9,))):
        cand = []
        with np.errstate(over="ignore", invalid="ignore"):     # 発散する候補もあるので警告を止める
            for beta in betas:
                for lr in grid:
                    k = steps_to(run_opt(kind, lr, 3000, beta=beta), 1e-6)
                    if k is not None:
                        cand.append((k, lr, beta))
        k, lr, beta = min(cand)
        best[kind] = (lr, beta)
        extra = f"、β = {beta}" if kind == "momentum" else ""
        print(f"    {kind:8s}: 学習率 {lr:.3f}{extra} のとき {k} 歩")
    lr_gd_fast = 0.075
    print(f"    （勾配降下法で学習率を 2/25 = 0.08 に近い {lr_gd_fast} にすると、θ2 の向きで行ったり来たりする）")

    print("\n(5) ミニバッチの勾配と、全データの勾配の差（2 → 64 → 64 → 2 のネット、渦巻き 2000 点、各 B で 300 回）")
    xb, yb = make_spiral(np.random.default_rng(5), 1000)
    mlp = MLP([2, 64, 64, 2], np.random.default_rng(0))
    full = np.concatenate([g.ravel() for g in mlp.loss_and_grads(xb, yb)[1]])
    r = np.random.default_rng(6)
    prev = None
    for B in (1, 4, 16, 64, 256):
        errs = []
        for _ in range(300):
            idx = r.choice(len(xb), B, replace=False)
            gB = np.concatenate([g.ravel() for g in mlp.loss_and_grads(xb[idx], yb[idx])[1]])
            errs.append(np.linalg.norm(gB - full) / np.linalg.norm(full))
        e = float(np.mean(errs))
        note = f"（B を 4 倍にして {prev / e:.2f} 分の 1）" if prev else ""
        print(f"    B = {B:3d}: 相対誤差の平均 {e:.3f}{note}")
        prev = e

    print("\n(6) バッチ正規化と層正規化（4 個のデータ × 3 個の特徴）")
    X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 9.0], [0.0, 1.0, 2.0], [5.0, 3.0, 4.0]])
    bn = (X - X.mean(0)) / X.std(0)
    ln = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    print(f"    バッチ正規化（列ごと）: 列の平均 {np.round(bn.mean(0), 12).tolist()}、列の標準偏差 {np.round(bn.std(0), 12).tolist()}")
    print(f"    層正規化（行ごと）    : 行の平均 {np.round(ln.mean(1), 12).tolist()}、行の標準偏差 {np.round(ln.std(1), 12).tolist()}")
    print("    1 行目だけを見ると: 入力 [1, 2, 3] → 層正規化 "
          f"{np.round(ln[0], 3).tolist()}（他の行に関係なく決まる）、バッチ正規化 {np.round(bn[0], 3).tolist()}")

    # ---------------------------------------------------------------- 図
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, act, cs, names in ((axes[0], "tanh", (0.5, 1.0, 2.0), ("0.5", "1（Xavier）", "2")),
                               (axes[1], "relu", (1.0, np.sqrt(2), 2.0), ("1", "√2（He）", "2"))):
        for c, nm, col in zip(cs, names, (C_A, C_B, C_C)):
            ax.plot(init_rows[(act, c)], color=col, lw=2, label=f"c = {nm}")
        ax.set_yscale("log")
        ax.set_xlabel("層")
        ax.set_title("tanh" if act == "tanh" else "ReLU", loc="left")
        ax.legend(fontsize=8.5, loc="lower left")
    axes[0].set_ylabel("値の標準偏差")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_5_init.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    styles = [(C_B, "o-"), (C_C, "^-"), (C_A, "s-"), (C_INK, "D-")]
    for (label, norms), (col, st) in zip(grad_rows.items(), styles):
        ax.plot(range(1, 21), norms, st, color=col, lw=1.8, ms=4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("層（1 が入力側、20 が出力側）")
    ax.set_ylabel("重みの勾配の大きさ")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_5_gradflow.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    colors = {"シグモイド": C_B, "ReLU（He）": C_A, "ReLU + 残差 + LN": C_INK}
    for (depth, label, lr), hist in train_rows.items():
        ax = axes[0] if depth == 20 else axes[1]
        ls = "-" if (depth == 20 or lr == 0.003) else "--"
        name = label if depth == 20 else f"{label}、学習率 {lr}"
        ax.plot(np.maximum(hist, 1e-5), color=colors[label], lw=1.8, ls=ls, label=name)
    for ax, depth in zip(axes, (20, 100)):
        ax.set_yscale("log")
        ax.set_xlabel("歩数")
        ax.set_title(f"{depth} 層", loc="left")
    axes[0].set_ylabel("損失")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / "ch03_5_deep_train.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    g1, g2 = np.meshgrid(np.linspace(-4.6, 1.6, 200), np.linspace(-2.2, 1.8, 200))
    axes[0].contour(g1, g2, 0.5 * (g1 ** 2 + 25 * g2 ** 2), levels=np.geomspace(0.05, 60, 9),
                    colors=[C_MUTED], linewidths=0.6)
    runs = [("勾配降下法", "gd", (lr_gd_fast, 0.0), C_B), ("モーメンタム", "momentum", best["momentum"], C_A),
            ("Adam", "adam", best["adam"], C_C)]
    for name, kind, (lr, beta), col in runs:
        path = run_opt(kind, lr, 40, beta=beta if kind == "momentum" else 0.9)
        axes[0].plot(path[:, 0], path[:, 1], "-o", color=col, lw=1.2, ms=2.5, label=name)
        full_path = run_opt(kind, lr, 180, beta=beta if kind == "momentum" else 0.9)
        vals = np.array([quad(p) for p in full_path])
        vals[vals < 1e-12] = np.nan                       # 図の下端より小さい部分は描かない
        axes[1].plot(vals, color=col, lw=2, label=name)
    axes[0].plot([0], [0], "*", color=C_INK, ms=10)
    axes[0].set_xlabel("θ1")
    axes[0].set_ylabel("θ2")
    axes[0].set_title("最初の 40 歩", loc="left")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-12, 1e3)
    axes[1].axhline(1e-6, color=C_MUTED, lw=0.8, ls=":")
    axes[1].set_xlabel("歩数")
    axes[1].set_title(r"f(θ)（点線は $10^{-6}$）", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_5_optim.png")
    plt.close(fig)
    saved("ch03_5_init.png", "ch03_5_gradflow.png", "ch03_5_deep_train.png", "ch03_5_optim.png")


if __name__ == "__main__":
    main()
