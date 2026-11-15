"""演習 3.3: 層を重ねると何ができるか

実行: python3 vla_math/ch03_3_expressivity.py
出力: docs/vla-math/figures/ch03_3_xor.png, ch03_3_activations.png, ch03_3_relu_approx.png,
      ch03_3_depth.png, ch03_3_hidden.png
必要: numpy, matplotlib（GPU は不要. 30 秒ほど）

(1) しきい値ニューロンを 2 層に並べ、重みを手で入れて XOR を作る. 隠れ層で点の位置が変わる様子を見る
(2) 同じ形のネット（隠れ層 H 個）を、乱数の初期値から勾配降下法で学習させる. H が小さいと失敗することがある
(3) 活性化関数がないと、何層重ねても 1 層の線形変換と同じになる
(4) 活性化関数とその微分の値
(5) ReLU を k 個足して、1 変数の関数を折れ線で近づける. k を増やすと誤差が減る
(6) 同じ数の ReLU でも、縦に重ねると折れ線の本数が指数的に増える（テント写像の合成）
(7) 渦巻きを分けるネットの最後の隠れ層（2 個）を見ると、点が直線で分けられる配置に並び替えられている
"""

from __future__ import annotations

import math

import numpy as np

from ch03_mlp import make_spiral, softmax
from common import C_A, C_B, C_C, C_INK, C_MUTED, blue_map, saved, setup_matplotlib, FIG_DIR

X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR = np.array([0, 1, 1, 0])


def step(z):
    return (z >= 0).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- (1) 手で作る XOR
W1_HAND = np.array([[1.0, 1.0], [1.0, 1.0]])   # 列が隠れニューロン. h1 = OR, h2 = AND
B1_HAND = np.array([-0.5, -1.5])
W2_HAND = np.array([1.0, -1.0])                # y = 1[h1 - h2 - 0.5 ≥ 0]
B2_HAND = -0.5


def xor_by_hand(x):
    h = step(x @ W1_HAND + B1_HAND)
    return h, step(h @ W2_HAND + B2_HAND)


# ---------------------------------------------------------------- (2) XOR を学習させる
def train_xor(H, seeds, steps=5000, lr=1.0):
    """2 → H（tanh）→ 1（シグモイド）. 交差エントロピーを、4 点まとめた勾配降下法で小さくする.

    初期値の違う len(seeds) 個のネットを、配列の先頭の軸に並べて同時に学習させる（1 個ずつ回すのと同じ計算）.
    """
    W1, W2 = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        W1.append(rng.normal(0, 1, (2, H)))
        W2.append(rng.normal(0, 1, (H, 1)))
    W1, W2 = np.stack(W1), np.stack(W2)               # (S, 2, H), (S, H, 1)
    b1, b2 = np.zeros((len(seeds), 1, H)), np.zeros((len(seeds), 1, 1))
    t = XOR.reshape(1, -1, 1).astype(float)
    for _ in range(steps):
        h = np.tanh(np.einsum("nd,sdh->snh", X4, W1) + b1)
        p = sigmoid(h @ W2 + b2)
        d2 = (p - t) / len(X4)                   # 出力側の誤差（2 章と同じ形）
        dh = (d2 @ W2.transpose(0, 2, 1)) * (1 - h ** 2)   # 隠れ層へ戻した誤差
        W2 -= lr * h.transpose(0, 2, 1) @ d2
        b2 -= lr * d2.sum(1, keepdims=True)
        W1 -= lr * np.einsum("nd,snh->sdh", X4, dh)
        b1 -= lr * dh.sum(1, keepdims=True)
    h = np.tanh(np.einsum("nd,sdh->snh", X4, W1) + b1)
    return sigmoid(h @ W2 + b2)[:, :, 0]         # (S, 4)


# ---------------------------------------------------------------- (4) 活性化関数
ERF = np.vectorize(math.erf)


def gelu(x):
    return x * 0.5 * (1 + ERF(x / math.sqrt(2)))


def gelu_grad(x):
    return 0.5 * (1 + ERF(x / math.sqrt(2))) + x * np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)


ACTS = [
    ("シグモイド", sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
    ("tanh", np.tanh, lambda x: 1 - np.tanh(x) ** 2),
    ("ReLU", lambda x: np.maximum(x, 0), lambda x: (x > 0).astype(float)),
    ("GELU", gelu, gelu_grad),
]


# ---------------------------------------------------------------- (5) ReLU で折れ線を作る
def target(x):
    return np.sin(2 * np.pi * x) + 0.4 * np.sin(5 * np.pi * x)


def relu_interpolant(k):
    """[0, 1] を k 等分した折れ線を、ReLU k 個と定数で表す. 戻り値: (折れ線の関数, 各項の係数, 折れ目)."""
    knots = np.linspace(0, 1, k + 1)
    vals = target(knots)
    slopes = np.diff(vals) / np.diff(knots)
    coef = np.concatenate([[slopes[0]], np.diff(slopes)])   # 折れ目ごとの「傾きの変化」
    bias = vals[0]

    def g(x):
        return bias + np.maximum(x[:, None] - knots[None, :-1], 0) @ coef

    return g, coef, knots[:-1], bias


# ---------------------------------------------------------------- (6) 深さ: テント写像
def tent(x):
    """ReLU 2 個でできる山形: T(x) = 2 ReLU(x) − 4 ReLU(x − 1/2)（0 ≤ x ≤ 1 で 0 → 1 → 0）."""
    return 2 * np.maximum(x, 0) - 4 * np.maximum(x - 0.5, 0)


def count_pieces(y, x):
    """折れ線の線分の数を、傾きが変わる回数から数える."""
    s = np.diff(y) / np.diff(x)
    return 1 + int(np.sum(np.abs(np.diff(s)) > 1e-6 * (1 + np.abs(s[:-1]))))


# ---------------------------------------------------------------- (7) 渦巻きの隠れ層を見る
class Net:
    """2 → 32（ReLU）→ 32（ReLU）→ 2（tanh）→ 2（softmax）. 逆伝播は 3.4 節の形をそのまま書く."""

    def __init__(self, rng):
        sizes = [2, 32, 32, 2, 2]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[2] *= np.sqrt(0.5)                 # tanh の層は分散 1/a（Xavier に近い）
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        a1 = np.maximum(x @ self.W[0] + self.b[0], 0)
        a2 = np.maximum(a1 @ self.W[1] + self.b[1], 0)
        h = np.tanh(a2 @ self.W[2] + self.b[2])
        logits = h @ self.W[3] + self.b[3]
        return a1, a2, h, logits

    def step(self, x, y, lr):
        a1, a2, h, logits = self.forward(x)
        p = softmax(logits)
        n = len(x)
        d4 = p.copy()
        d4[np.arange(n), y] -= 1
        d4 /= n
        d3 = (d4 @ self.W[3].T) * (1 - h ** 2)
        d2 = (d3 @ self.W[2].T) * (a2 > 0)
        d1 = (d2 @ self.W[1].T) * (a1 > 0)
        gW = [x.T @ d1, a1.T @ d2, a2.T @ d3, h.T @ d4]
        gb = [d1.sum(0), d2.sum(0), d3.sum(0), d4.sum(0)]
        self.t += 1
        for i, (prm, g) in enumerate(zip(self.W + self.b, gW + gb)):   # Adam（3.5 節）
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            prm -= lr * mh / (np.sqrt(vh) + 1e-8)
        return -np.mean(np.log(p[np.arange(n), y] + 1e-12))


def main():
    plt = setup_matplotlib()

    print("(1) しきい値ニューロン 3 個（隠れ層 2 個 + 出力 1 個）で XOR を作る")
    h, y = xor_by_hand(X4)
    for xi, hi, yi in zip(X4.astype(int), h.astype(int), y.astype(int)):
        print(f"    入力 {xi.tolist()} → 隠れ層 (h1, h2) = {hi.tolist()} → 出力 {yi}")
    print(f"    出力 {y.astype(int).tolist()}、XOR の正解 {XOR.tolist()}")

    print("\n(2) 2 → H → 1 のネットを、乱数の初期値から学習させる（各 H で初期値 100 通り、5000 歩）")
    for H in (1, 2, 3, 4, 8):
        p = train_xor(H, range(100))
        ok = int(np.sum(np.all((p >= 0.5) == (XOR == 1), axis=1)))
        print(f"    隠れ層 {H} 個: 4 点すべて正解になったのは {ok} / 100 回")

    print("\n(3) 活性化関数がないと、2 層は 1 層と同じ")
    rng = np.random.default_rng(0)
    W1, b1 = rng.normal(size=(2, 5)), rng.normal(size=5)
    W2, b2 = rng.normal(size=(5, 1)), rng.normal(size=1)
    x = rng.normal(size=(1000, 2))
    two = (x @ W1 + b1) @ W2 + b2
    one = x @ (W1 @ W2) + (b1 @ W2 + b2)
    print(f"    (x W1 + b1) W2 + b2 と x (W1 W2) + (b1 W2 + b2) の差の最大: {np.max(np.abs(two - one)):.1e}")

    print("\n(4) 活性化関数の微分の最大値")
    z = np.linspace(-6, 6, 120001)
    for name, f, df in ACTS:
        d = df(z)
        print(f"    {name:6s}: 微分の最大 {d.max():.3f}（z = {z[np.argmax(d)]:+.2f} のとき）")

    print("\n(5) ReLU k 個の折れ線で、f(x) = sin(2πx) + 0.4 sin(5πx) を近づける（0 ≤ x ≤ 1）")
    xs = np.linspace(0, 1, 20001)
    errs = {}
    for k in (2, 4, 8, 16, 32, 64, 128):
        g = relu_interpolant(k)[0]
        errs[k] = np.max(np.abs(g(xs) - target(xs)))
        ratio = f"（前の {errs[k // 2] / errs[k]:.1f} 分の 1）" if k // 2 in errs else ""
        print(f"    k = {k:3d}: 誤差の最大 {errs[k]:.4f}{ratio}")

    print("\n(6) テント写像 T を L 回合成する（1 回の T に ReLU 2 個）")
    xs6 = np.linspace(0, 1, 2 ** 12 + 1)          # 折れ目 (j / 2^L) をすべて含む格子
    ys = xs6.copy()
    rows = []
    for L in range(1, 9):
        ys = tent(ys)
        pieces = count_pieces(ys, xs6)
        rows.append((L, pieces))
        print(f"    L = {L}: ReLU {2 * L:2d} 個（2 個 × {L} 層）で線分 {pieces:3d} 本. "
              f"隠れ層 1 つで同じ本数を作るには ReLU が {pieces - 1} 個以上必要")

    print("\n(7) 渦巻きを、2 → 32 → 32 → 2（tanh）→ 2 のネットで分ける")
    rng = np.random.default_rng(0)
    xs7, ys7 = make_spiral(rng, 200)
    net = Net(np.random.default_rng(1))
    snaps = {}
    for it in range(3001):
        if it in (0, 25, 50, 100, 3000):
            _, _, hid, logits = net.forward(xs7)
            p = softmax(logits)
            loss = -np.mean(np.log(p[np.arange(len(ys7)), ys7]))
            acc = np.mean(logits.argmax(1) == ys7)
            snaps[it] = (hid.copy(), acc)
            print(f"    {it:4d} 歩: 損失 {loss:.2e}、正解率 {100 * acc:.1f}%")
        net.step(xs7, ys7, lr=0.01)

    # ---------------------------------------------------------------- 図
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    ax = axes[0]
    xx = np.linspace(-0.4, 1.4, 10)
    ax.plot(xx, 0.5 - xx, color=C_INK, lw=1.4, label="h1 の境目（OR）")
    ax.plot(xx, 1.5 - xx, color=C_INK, lw=1.4, ls="--", label="h2 の境目（AND）")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("入力の空間", loc="left")
    ax2 = axes[1]
    xx2 = np.linspace(-0.4, 1.4, 10)
    ax2.plot(xx2, xx2 - 0.5, color=C_MUTED, lw=1.6, label="出力の境目")
    ax2.text(1.0, -0.2, "2 点が重なる", fontsize=8.5, color=C_INK, ha="center", va="top")
    ax2.set_xlabel("h1")
    ax2.set_ylabel("h2")
    ax2.set_title("隠れ層の空間", loc="left")
    for a, pts in ((ax, X4), (ax2, h)):
        for k, (c, mk, lab) in enumerate(((C_B, "o", "XOR = 0"), (C_A, "s", "XOR = 1"))):
            m = XOR == k
            a.scatter(pts[m, 0], pts[m, 1], s=110, color=c, marker=mk, zorder=3, label=lab,
                      edgecolors="white", linewidths=1.0)
        a.set_xlim(-0.4, 1.4)
        a.set_ylim(-0.45, 1.4)
        a.set_xticks([0, 1])
        a.set_yticks([0, 1])
        a.set_aspect("equal")
    hnd, lab = ax.get_legend_handles_labels()
    hnd2, lab2 = ax2.get_legend_handles_labels()
    order = [2, 3, 0, 1]
    fig.legend([hnd[i] for i in order] + [hnd2[0]], [lab[i] for i in order] + [lab2[0]],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIG_DIR / "ch03_3_xor.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.6), sharex=True)
    zz = np.linspace(-4, 4, 801)
    for ax, (name, f, df) in zip(axes.ravel(), ACTS):
        if name == "シグモイド":
            ax.plot(zz, step(zz), color=C_MUTED, lw=1.2, ls=":", label="ステップ")
        ax.plot(zz, f(zz), color=C_A, lw=2, label="関数")
        ax.plot(zz, df(zz), color=C_B, lw=2, ls="--", label="微分")
        ax.axhline(0, color=C_MUTED, lw=0.6)
        ax.set_title(name, loc="left")
        ax.set_ylim(-1.15, 1.6 if name in ("ReLU", "GELU") else 1.15)
    axes[0, 0].legend(loc="upper left", fontsize=8.5, handlelength=1.6)
    for ax in axes[1]:
        ax.set_xlabel("z")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_3_activations.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    g4, coef4, knots4, bias4 = relu_interpolant(4)
    ax = axes[0]
    ax.plot(xs, target(xs), color=C_MUTED, lw=1.2, ls=":")
    shades = blue_map()(np.linspace(0.35, 1.0, 4))
    part = np.full_like(xs, bias4)
    for j, (c, kn) in enumerate(zip(coef4, knots4)):
        part = part + c * np.maximum(xs - kn, 0)
        ax.plot(xs, part, color=shades[j], lw=1.2 + 0.4 * j, label=f"{j + 1} 個")
    ax.set_title("ReLU を 1 個ずつ足す", loc="left", fontsize=10.5)
    ax.set_ylim(-1.6, 2.6)
    ax.set_xlabel("x")
    ax.legend(fontsize=8.5, loc="upper right", handlelength=1.4, ncol=2)
    ax = axes[1]
    ax.plot(xs, target(xs), color=C_MUTED, lw=3.2, label="目標", alpha=0.6)
    for k, c in zip((4, 8, 16), (C_A, C_B, C_C)):
        ax.plot(xs, relu_interpolant(k)[0](xs), color=c, lw=1.4, label=f"k = {k}")
    ax.set_title("k を増やすと目標に近づく", loc="left", fontsize=10.5)
    ax.set_xlabel("x")
    ax.legend(fontsize=8.5, loc="lower left", handlelength=1.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_3_relu_approx.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15), sharey=True)
    ys = xs6.copy()
    for L, ax in zip(range(1, 5), axes):
        ys = tent(ys)
        ax.plot(xs6, ys, color=C_A, lw=1.4)
        ax.set_title(f"{L} 層: {2 ** L} 本", loc="left", fontsize=10.5)
        ax.set_xticks([0, 1])
        ax.set_xlabel("x")
    axes[0].set_yticks([0, 1])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_3_depth.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6))
    panels = [("入力 (x1, x2)", xs7, None)] + [
        (f"隠れ層 {it} 歩（{100 * snaps[it][1]:.0f}%）", snaps[it][0], it) for it in (0, 50, 3000)]
    for ax, (title, pts, it) in zip(axes.ravel(), panels):
        for k, (c, mk) in enumerate(((C_A, "o"), (C_B, "s"))):
            m = ys7 == k
            ax.scatter(pts[m, 0], pts[m, 1], s=6, color=c, marker=mk, linewidths=0)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_aspect("equal")
        if it is not None:
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel("h1")
            ax.set_ylabel("h2")
    w = net.W[3][:, 1] - net.W[3][:, 0]          # 最後の層の境目: (h1, h2) の 1 次式 = 0
    b = net.b[3][1] - net.b[3][0]
    hx = np.linspace(-1.1, 1.1, 10)
    if abs(w[1]) > 1e-9:
        axes[1, 1].plot(hx, -(w[0] * hx + b) / w[1], color=C_INK, lw=1.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_3_hidden.png")
    plt.close(fig)
    saved("ch03_3_xor.png", "ch03_3_activations.png", "ch03_3_relu_approx.png", "ch03_3_depth.png",
          "ch03_3_hidden.png")


if __name__ == "__main__":
    main()
