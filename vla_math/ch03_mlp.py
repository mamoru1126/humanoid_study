"""演習 3: ニューラルネットと逆伝播

実行: python3 vla_math/ch03_mlp.py
出力: docs/vla-math/figures/ch03_boundary.png
必要: numpy, matplotlib（GPU は不要. 10 秒ほど）

2 本の渦巻きに沿って並んだ点を、2 つのクラスに分ける.
(1) 線形のモデル（2 章の softmax 回帰）では、ほとんど分けられない
(2) 隠れ層 2 つのニューラルネット（重みは numpy で持ち、逆伝播も手で書く）なら分けられる
(3) 逆伝播で求めた勾配が、数値微分と一致することを確かめる
(4) 隠れ層の幅を変えて、表せる形がどう変わるかを見る
"""

from __future__ import annotations

import numpy as np

from common import C_A, C_B, saved, setup_matplotlib, FIG_DIR


def make_spiral(rng, n_per_class, noise=0.05):
    xs, ys = [], []
    for k in range(2):
        t = np.sqrt(rng.uniform(0.02, 1.0, n_per_class))      # 中心から外へ
        ang = 3.0 * np.pi * t + k * np.pi
        r = t
        p = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + noise * rng.standard_normal((n_per_class, 2))
        xs.append(p)
        ys.append(np.full(n_per_class, k))
    return np.concatenate(xs), np.concatenate(ys)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """全結合の層を重ねたネット. 隠れ層の活性化関数は ReLU、出力はクラスごとのロジット."""

    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in))   # He の初期化
            self.b.append(np.zeros(n_out))
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return self.W + self.b

    def n_params(self):
        return sum(p.size for p in self.params())

    def forward(self, x):
        """順伝播. 逆伝播で使う途中の値を取っておく."""
        self.hs = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            h = a if i == len(self.W) - 1 else np.maximum(a, 0.0)
            self.hs.append(h)
        return h

    def loss_and_grads(self, x, y):
        """平均の交差エントロピーと、全部の重みについての勾配（逆伝播）."""
        n = len(x)
        z = self.forward(x)
        p = softmax(z)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
        g = p.copy()
        g[np.arange(n), y] -= 1.0
        g /= n                                         # 出力についての勾配: 予測 - 正解
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = self.hs[i].T @ g                    # 重みの勾配 = 入力ᵀ × 出力側の勾配
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = g @ self.W[i].T                     # 1 つ前の層の出力へ戻す
                g = g * (self.hs[i] > 0)                # ReLU の微分（正なら 1、負なら 0）
        return loss, gW + gb

    def adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1**self.t)) / (np.sqrt(v / (1 - b2**self.t)) + eps)


def train(net, x, y, steps, lr):
    for _ in range(steps):
        loss, grads = net.loss_and_grads(x, y)
        net.adam_step(grads, lr)
    return loss


def accuracy(net, x, y):
    return float(np.mean(np.argmax(net.forward(x), axis=1) == y))


def main():
    rng = np.random.default_rng(0)
    x_tr, y_tr = make_spiral(rng, 200)
    x_va, y_va = make_spiral(rng, 500)

    print("(1) 線形のモデル（入力 2 → 出力 2）")
    lin = MLP([2, 2], rng)
    loss = train(lin, x_tr, y_tr, 2000, 0.05)
    print(f"    重み {lin.n_params()} 個、学習後の交差エントロピー {loss:.3f}、検証データの正解率 {100 * accuracy(lin, x_va, y_va):.1f}%")

    print("\n(2) 隠れ層 2 つのニューラルネット（2 → 64 → 64 → 2、ReLU）")
    net = MLP([2, 64, 64, 2], rng)
    for k, steps in enumerate((0, 200, 800, 2000)):
        if steps:
            loss = train(net, x_tr, y_tr, steps, 0.01)
        else:
            loss = net.loss_and_grads(x_tr, y_tr)[0]
        done = sum((0, 200, 800, 2000)[: k + 1])
        print(f"    {done:4d} 回目: 交差エントロピー {loss:.3f}、検証データの正解率 {100 * accuracy(net, x_va, y_va):.1f}%")
    print(f"    重みの数 {net.n_params()} 個")

    print("\n(3) 勾配の確認（逆伝播 と 数値微分）")
    chk = MLP([2, 16, 16, 2], np.random.default_rng(1))
    _, grads = chk.loss_and_grads(x_tr, y_tr)
    errs = []
    h = 1e-6        # ReLU の折れ目をまたぐと差分がずれるので、小さめに取る
    rng_c = np.random.default_rng(2)
    for pi, p in enumerate(chk.params()):
        for _ in range(5):
            idx = tuple(rng_c.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + h
            lp = chk.loss_and_grads(x_tr, y_tr)[0]
            p[idx] = old - h
            lm = chk.loss_and_grads(x_tr, y_tr)[0]
            p[idx] = old
            num = (lp - lm) / (2 * h)
            errs.append(abs(num - grads[pi][idx]) / max(1e-8, abs(num) + abs(grads[pi][idx])))
    print(f"    30 か所で比べた相対誤差: 最大 {max(errs):.1e}、中央値 {np.median(errs):.1e}")

    print("\n(4) 隠れ層の幅を変える（2 → 幅 → 幅 → 2、各 3000 回）")
    nets = {}
    for width in (2, 4, 8, 64):
        nw = MLP([2, width, width, 2], np.random.default_rng(3))
        train(nw, x_tr, y_tr, 3000, 0.01)
        nets[width] = nw
        print(f"    幅 {width:3d}: 重み {nw.n_params():5d} 個、検証データの正解率 {100 * accuracy(nw, x_va, y_va):.1f}%")

    plt = setup_matplotlib()
    g = np.linspace(-1.25, 1.25, 260)
    gx, gy = np.meshgrid(g, g)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.9))
    panels = [(lin, "線形のモデル"), (nets[4], "隠れ層の幅 4"), (nets[8], "隠れ層の幅 8"), (net, "隠れ層の幅 64")]
    for ax, (model, title) in zip(axes.ravel(), panels):
        cls = np.argmax(model.forward(grid), axis=1).reshape(gx.shape)
        ax.contourf(gx, gy, cls, levels=[-0.5, 0.5, 1.5], colors=["#E8EAF6", "#FBE9E7"])
        for k, (c, mk) in enumerate(((C_A, "o"), (C_B, "s"))):
            m = y_tr == k
            ax.scatter(x_tr[m, 0], x_tr[m, 1], s=6, color=c, marker=mk, linewidths=0)
        ax.set_title(f"{title}　{100 * accuracy(model, x_va, y_va):.0f}%", fontsize=10.5, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_boundary.png")
    plt.close(fig)
    saved("ch03_boundary.png")


if __name__ == "__main__":
    main()
