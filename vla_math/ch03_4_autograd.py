"""演習 3.4: 逆伝播を自分で書く（自動微分）

実行: python3 vla_math/ch03_4_autograd.py
出力: docs/vla-math/figures/ch03_4_xor.png, ch03_4_cost.png
必要: numpy, matplotlib（GPU は不要. 30 秒ほど）

(1) スカラー 1 つを入れる箱 Value を作る. 足し算や掛け算のたびに「誰から作られたか」を覚えておき、
    最後に backward() を呼ぶと、出口から入口へ向かって連鎖律で勾配を配る（約 50 行）
(2) 1 個のニューロンと 2 乗誤差で、計算グラフの各節点の値と勾配を確かめる（3.4 節の図の数字）
(3) 同じ変数を 2 回使うと、勾配は足し合わせになる
(4) Value で 2 → 4 → 1 のネットを組み、XOR を学習させる. 勾配を数値微分・行列の式と比べる
(5) 行列の層 Y = X W + b の勾配の式を、Value で 1 要素ずつ計算した結果と比べる
(6) 数値微分と逆伝播の計算時間を、パラメータの数を変えて比べる
"""

from __future__ import annotations

import math
import time

import numpy as np

from ch03_mlp import MLP, make_spiral, softmax
from common import C_A, C_B, C_INK, C_MUTED, saved, setup_matplotlib, FIG_DIR, blue_map


# ---------------------------------------------------------------- (1) 自動微分の本体
class Value:
    """スカラー 1 つと、その勾配を持つ箱. 演算の結果は、材料（parents）と、勾配の配り方（_back）を覚える."""

    def __init__(self, data, parents=(), op=""):
        self.data, self.grad = float(data), 0.0
        self.parents, self.op = parents, op
        self._back = lambda: None            # 自分の勾配を、材料へ配る関数

    @staticmethod
    def wrap(v):
        return v if isinstance(v, Value) else Value(v)

    def __add__(self, other):
        other = Value.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def back():                           # 足し算: 勾配をそのまま両方へ
            self.grad += out.grad
            other.grad += out.grad
        out._back = back
        return out

    def __mul__(self, other):
        other = Value.wrap(other)
        out = Value(self.data * other.data, (self, other), "×")

        def back():                           # 掛け算: 相手の値を掛けて配る
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._back = back
        return out

    def __pow__(self, k):                     # k は普通の数
        out = Value(self.data ** k, (self,), f"^{k}")

        def back():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._back = back
        return out

    def _unary(self, value, local_grad, op):
        """1 入力の関数. local_grad は「出力を入力で微分した値」."""
        out = Value(value, (self,), op)

        def back():
            self.grad += local_grad * out.grad
        out._back = back
        return out

    def tanh(self):
        t = math.tanh(self.data)
        return self._unary(t, 1 - t * t, "tanh")

    def exp(self):
        e = math.exp(self.data)
        return self._unary(e, e, "exp")

    def log(self):
        return self._unary(math.log(self.data), 1 / self.data, "log")

    def relu(self):
        return self._unary(max(self.data, 0.0), float(self.data > 0), "ReLU")

    def backward(self):
        """出口（self）の勾配を 1 として、作られた順の逆に、各節点の _back を 1 回ずつ呼ぶ."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:                          # 深さ優先で「材料が先に来る」並び（トポロジカル順）を作る
            v, done = stack.pop()
            if done:
                order.append(v)
            elif id(v) not in seen:
                seen.add(id(v))
                stack.append((v, True))
                stack.extend((p, False) for p in v.parents)
        self.grad = 1.0
        for v in reversed(order):
            v._back()

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value.wrap(other))

    def __rsub__(self, other):
        return Value.wrap(other) + (-self)

    def __truediv__(self, other):
        return self * Value.wrap(other) ** -1


def sigmoid(z):
    return (1 + (-z).exp()) ** -1


# ---------------------------------------------------------------- (4) Value でネットを組む
class TinyNet:
    """2 → H（tanh）→ 1（シグモイド）. 重みはすべて Value."""

    def __init__(self, H, rng):
        self.W1 = [[Value(rng.normal(0, 1)) for _ in range(H)] for _ in range(2)]
        self.b1 = [Value(0.0) for _ in range(H)]
        self.W2 = [Value(rng.normal(0, 1)) for _ in range(H)]
        self.b2 = Value(0.0)

    def params(self):
        return [w for row in self.W1 for w in row] + self.b1 + self.W2 + [self.b2]

    def __call__(self, x):
        H = len(self.b1)
        h = [(x[0] * self.W1[0][j] + x[1] * self.W1[1][j] + self.b1[j]).tanh() for j in range(H)]
        z = self.b2
        for j in range(H):
            z = z + h[j] * self.W2[j]
        return sigmoid(z)

    def loss(self, X, T):
        total = Value(0.0)
        for x, t in zip(X, T):
            p = self(x)
            total = total - (t * p.log() + (1 - t) * (1 - p).log())   # 交差エントロピー
        return total / len(X)

    def numpy_params(self):
        H = len(self.b1)
        W1 = np.array([[w.data for w in row] for row in self.W1])
        b1 = np.array([b.data for b in self.b1])
        W2 = np.array([w.data for w in self.W2])
        return W1, b1, W2, self.b2.data, H


def numpy_forward_backward(W1, b1, W2, b2, X, T):
    """同じネットを行列で書いた順伝播と逆伝播（3.4 節の式）."""
    h = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(h @ W2 + b2)))
    loss = -np.mean(T * np.log(p) + (1 - T) * np.log(1 - p))
    dz = (p - T) / len(X)
    dW2, db2 = h.T @ dz, dz.sum()
    da = np.outer(dz, W2) * (1 - h ** 2)
    return loss, np.concatenate([(X.T @ da).ravel(), da.sum(0), dW2, [db2]])


def main():
    print("(2) 1 個のニューロンと 2 乗誤差: L = (w x + b − y)²")
    w, x, b, y = Value(2.0), Value(3.0), Value(1.0), Value(5.0)
    u = w * x
    z = u + b
    e = z - y
    L = e ** 2
    L.backward()
    for name, v in (("w", w), ("x", x), ("b", b), ("y", y), ("u = w x", u), ("z = u + b", z),
                    ("e = z − y", e), ("L = e²", L)):
        print(f"    {name:10s} 値 {v.data:5.1f}   ∂L/∂(この節点) = {v.grad:5.1f}")

    def L_of(wv, xv, bv, yv):
        return (wv * xv + bv - yv) ** 2
    h = 1e-6
    num = [(L_of(2 + h, 3, 1, 5) - L_of(2 - h, 3, 1, 5)) / (2 * h),
           (L_of(2, 3 + h, 1, 5) - L_of(2, 3 - h, 1, 5)) / (2 * h),
           (L_of(2, 3, 1 + h, 5) - L_of(2, 3, 1 - h, 5)) / (2 * h)]
    print(f"    数値微分: ∂L/∂w ≈ {num[0]:.6f}, ∂L/∂x ≈ {num[1]:.6f}, ∂L/∂b ≈ {num[2]:.6f}")

    print("\n(3) 同じ変数を 2 回使う: f = x y + x（x = 3, y = −2）")
    x, y = Value(3.0), Value(-2.0)
    f = x * y + x
    f.backward()
    print(f"    f = {f.data:.0f}, ∂f/∂x = {x.grad:.0f}（= y + 1. 2 本の道の勾配の和）, ∂f/∂y = {y.grad:.0f}")

    print("\n(4) Value で 2 → 4 → 1 のネットを組み、XOR を学習させる（4 点まとめた勾配降下法、学習率 0.5）")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    T = np.array([0, 1, 1, 0], dtype=float)
    net = TinyNet(4, np.random.default_rng(0))
    loss = net.loss(X.tolist(), T.tolist())
    loss.backward()
    g_engine = np.array([p.grad for p in net.params()])
    W1, b1, W2, b2, H = net.numpy_params()
    _, g_matrix = numpy_forward_backward(W1, b1, W2, b2, X, T)
    g_num = np.zeros_like(g_engine)
    for i, p in enumerate(net.params()):
        old = p.data
        p.data = old + 1e-6
        lp = net.loss(X.tolist(), T.tolist()).data
        p.data = old - 1e-6
        lm = net.loss(X.tolist(), T.tolist()).data
        p.data = old
        g_num[i] = (lp - lm) / 2e-6
    print(f"    パラメータ {len(g_engine)} 個. 学習前の勾配の差の最大: "
          f"Value と数値微分 {np.max(np.abs(g_engine - g_num)):.1e}、Value と行列の式 {np.max(np.abs(g_engine - g_matrix)):.1e}")
    n_nodes = 0
    stack, seen = [loss], set()
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        n_nodes += 1
        stack.extend(v.parents)
    print(f"    損失 1 回分の計算グラフの節点の数: {n_nodes}")
    results = {}
    for seed in (0, 1):
        net_s = TinyNet(4, np.random.default_rng(seed))
        history = []
        t0 = time.perf_counter()
        for it in range(2001):
            for p in net_s.params():
                p.grad = 0.0                  # 勾配は足し込むので、毎回 0 に戻す
            loss = net_s.loss(X.tolist(), T.tolist())
            loss.backward()
            history.append(loss.data)
            for p in net_s.params():
                p.data -= 0.5 * p.grad
        probs = [net_s(xi).data for xi in X.tolist()]
        results[seed] = (net_s, history)
        print(f"    初期値の乱数 {seed}: 損失 {history[0]:.4f} → {history[100]:.4f}（100 歩）→ "
              f"{history[500]:.4f}（500 歩）→ {history[-1]:.4f}（2000 歩）. {time.perf_counter() - t0:.1f} 秒")
        print(f"      出力の確率: {np.round(probs, 3).tolist()}（正解 {T.astype(int).tolist()}）")
    net, history = results[1]

    print("\n(5) 行列の層 Y = X W + b の勾配の式を、Value で 1 要素ずつ計算したものと比べる")
    rng = np.random.default_rng(3)
    Xn, Wn, bn, Gn = rng.normal(size=(3, 2)), rng.normal(size=(2, 4)), rng.normal(size=4), rng.normal(size=(3, 4))
    Xv = [[Value(v) for v in row] for row in Xn]
    Wv = [[Value(v) for v in row] for row in Wn]
    bv = [Value(v) for v in bn]
    S = Value(0.0)
    for i in range(3):
        for j in range(4):
            yij = bv[j] + sum((Xv[i][k] * Wv[k][j] for k in range(2)), Value(0.0))
            S = S + yij * Gn[i, j]            # S = Σ G_ij Y_ij なので ∂S/∂Y = G
    S.backward()
    dW = np.array([[v.grad for v in row] for row in Wv])
    db = np.array([v.grad for v in bv])
    dX = np.array([[v.grad for v in row] for row in Xv])
    print(f"    ∂S/∂W と Xᵀ G の差 {np.max(np.abs(dW - Xn.T @ Gn)):.1e}、∂S/∂b と G の列の和の差 "
          f"{np.max(np.abs(db - Gn.sum(0))):.1e}、∂S/∂X と G Wᵀ の差 {np.max(np.abs(dX - Gn @ Wn.T)):.1e}")

    print("\n(6) 数値微分と逆伝播の時間（2 → H → H → 2 の ReLU ネット、渦巻き 400 点、各 3 回の最短）")
    xs, ys = make_spiral(np.random.default_rng(0), 200)
    rows = []
    for H in (4, 8, 16, 32, 64):
        mlp = MLP([2, H, H, 2], np.random.default_rng(0))
        P = mlp.n_params()

        def loss_only():
            p_ = softmax(mlp.forward(xs))
            return -np.mean(np.log(p_[np.arange(len(ys)), ys] + 1e-12))

        t_bp = min(_timeit(lambda: mlp.loss_and_grads(xs, ys)) for _ in range(3))
        t_fw = min(_timeit(loss_only) for _ in range(3))

        def numeric():
            for prm in mlp.params():
                flat = prm.reshape(-1)
                for i in range(flat.size):
                    old = flat[i]
                    flat[i] = old + 1e-5
                    lp = loss_only()
                    flat[i] = old - 1e-5
                    lm = loss_only()
                    flat[i] = old
        t_num = _timeit(numeric)
        rows.append((P, t_fw, t_bp, t_num))
        print(f"    H = {H:2d}: パラメータ {P:5d} 個. 順伝播 1 回 {1e3 * t_fw:6.3f} ms、逆伝播込み {1e3 * t_bp:6.3f} ms"
              f"（順伝播の {t_bp / t_fw:.1f} 倍）、数値微分 {t_num:7.3f} s（逆伝播の {t_num / t_bp:,.0f} 倍）")

    # ---------------------------------------------------------------- 図
    plt = setup_matplotlib()
    W1, b1, W2, b2, H = net.numpy_params()
    g = np.linspace(-0.5, 1.5, 201)
    gx, gy = np.meshgrid(g, g)
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    pg = (1 / (1 + np.exp(-(np.tanh(G @ W1 + b1) @ W2 + b2)))).reshape(gx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].plot(history, color=C_A, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("歩数")
    axes[0].set_title("損失（交差エントロピー）", loc="left")
    im = axes[1].pcolormesh(gx, gy, pg, cmap=blue_map(), vmin=0, vmax=1, shading="auto", rasterized=True)
    axes[1].contour(gx, gy, pg, levels=[0.5], colors=[C_INK], linewidths=1.2)
    for k, (c, mk) in enumerate(((C_B, "o"), ("white", "s"))):
        m = T == k
        axes[1].scatter(X[m, 0], X[m, 1], s=80, color=c, marker=mk, edgecolors=C_INK, linewidths=1.0, zorder=3)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_title("出力の確率（線は 0.5）", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_4_xor.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    P = np.array([r[0] for r in rows])
    ax.plot(P, [r[3] for r in rows], "o-", color=C_B, lw=2, ms=6, label="数値微分（全パラメータ）")
    ax.plot(P, [r[2] for r in rows], "s-", color=C_A, lw=2, ms=6, label="逆伝播（順伝播込み）")
    ax.plot(P, [r[1] for r in rows], ":", color=C_MUTED, lw=1.5, label="順伝播 1 回")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("パラメータの数")
    ax.set_ylabel("時間（秒）")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_4_cost.png")
    plt.close(fig)
    saved("ch03_4_xor.png", "ch03_4_cost.png")


def _timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
