"""演習 2: 確率で考える（最尤推定、softmax、交差エントロピー、KL）

実行: python3 vla_math/ch02_softmax.py
出力: docs/vla-math/figures/ch02_regions.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

(1) 正規分布の最尤推定: 平均の推定値は、二乗誤差を最小にする値（標本平均）と一致する
(2) softmax 回帰: センサの値 2 つから、床の種類（3 種類）を当てる分類器を、交差エントロピーで学習する.
    逆伝播で求めた勾配「予測 - 正解」が、数値微分と一致することも確かめる
(3) 温度: 同じロジットでも、温度で確率の広がりが変わる
(4) 交差エントロピー = エントロピー + KL ダイバージェンス を数値で確かめる
"""

from __future__ import annotations

import numpy as np

from common import C_INK, SERIES, saved, setup_matplotlib, FIG_DIR

CLASSES = ["タイル", "カーペット", "木の床"]
CENTERS = np.array([[-1.0, -0.6], [1.1, -0.4], [0.0, 1.2]])


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=float) / temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_data(rng, n_per_class):
    xs, ys = [], []
    for k, c in enumerate(CENTERS):
        xs.append(c + 0.55 * rng.standard_normal((n_per_class, 2)))
        ys.append(np.full(n_per_class, k))
    return np.concatenate(xs), np.concatenate(ys)


def loss_and_grad(W, b, x, y):
    """平均の交差エントロピーと、その勾配. ロジット z = x W + b."""
    n = len(x)
    p = softmax(x @ W + b)
    loss = -np.mean(np.log(p[np.arange(n), y]))
    g = p.copy()
    g[np.arange(n), y] -= 1.0             # ロジットについての勾配 = 予測の確率 - 正解（one-hot）
    g /= n
    return loss, x.T @ g, g.sum(axis=0)


def main():
    rng = np.random.default_rng(0)

    print("(1) 正規分布の最尤推定")
    data = 2.0 + 0.5 * rng.standard_normal(1000)
    mus = np.linspace(1.5, 2.5, 100001)
    nll = [0.5 * np.mean((data - m) ** 2) / 0.25 for m in mus]   # σ=0.5 として、定数を除いた負の対数尤度
    mu_hat = mus[int(np.argmin(nll))]
    print(f"    負の対数尤度を最小にする平均 {mu_hat:.4f}、標本平均 {data.mean():.4f}")
    print(f"    （負の対数尤度は、二乗誤差の定数倍に定数を足したもの. 最小にする値は同じ）")

    print("\n(2) softmax 回帰で、床の種類を当てる")
    x_tr, y_tr = make_data(rng, 100)
    x_va, y_va = make_data(rng, 300)
    W = np.zeros((2, 3))
    b = np.zeros(3)
    # 勾配の確認: 逆伝播の式と、数値微分（中心差分）を比べる
    W_chk = 0.3 * rng.standard_normal((2, 3))
    b_chk = 0.1 * rng.standard_normal(3)
    _, gW, gb = loss_and_grad(W_chk, b_chk, x_tr, y_tr)
    num = np.zeros_like(W_chk)
    h = 1e-6
    for i in range(2):
        for j in range(3):
            Wp = W_chk.copy()
            Wp[i, j] += h
            Wm = W_chk.copy()
            Wm[i, j] -= h
            num[i, j] = (loss_and_grad(Wp, b_chk, x_tr, y_tr)[0] - loss_and_grad(Wm, b_chk, x_tr, y_tr)[0]) / (2 * h)
    rel = np.abs(num - gW).max() / np.abs(num).max()
    print(f"    勾配の確認: 式で求めた勾配と数値微分の相対誤差 {rel:.1e}")
    lr = 0.5
    for step in range(501):
        loss, gW, gb = loss_and_grad(W, b, x_tr, y_tr)
        if step in (0, 10, 100, 500):
            acc = np.mean(np.argmax(x_va @ W + b, axis=1) == y_va)
            print(f"    {step:4d} 回目: 交差エントロピー {loss:.3f}、検証データの正解率 {100 * acc:.1f}%")
        W -= lr * gW
        b -= lr * gb
    p_va = softmax(x_va @ W + b)
    conf = p_va.max(axis=1)
    pred = p_va.argmax(axis=1)
    for lo, hi in ((0.0, 0.6), (0.6, 0.9), (0.9, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"    確信度 {lo:.1f}〜{min(hi, 1.0):.1f}: {m.sum():3d} 点、そのうち正解 {100 * np.mean(pred[m] == y_va[m]):.0f}%")

    print("\n(3) 温度: ロジット z = (2, 1, 0) を softmax で確率にする")
    for t in (0.5, 1.0, 2.0, 10.0):
        p = softmax([2.0, 1.0, 0.0], t)
        print(f"    温度 {t:4.1f}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    print("\n(4) 交差エントロピー = エントロピー + KL ダイバージェンス")
    p = np.array([0.7, 0.2, 0.1])      # データの分布（正しい答え）
    q = np.array([0.5, 0.3, 0.2])      # モデルの分布
    ce = -np.sum(p * np.log(q))
    ent = -np.sum(p * np.log(p))
    kl = np.sum(p * np.log(p / q))
    print(f"    交差エントロピー {ce:.4f} = エントロピー {ent:.4f} + KL {kl:.4f}（足すと {ent + kl:.4f}）")
    print(f"    q を p に一致させると KL = {np.sum(p * np.log(p / p)):.4f}、交差エントロピーはエントロピーまで下がる")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    g = np.linspace(-3, 3, 300)
    gx, gy = np.meshgrid(g, g)
    pg = softmax(np.stack([gx.ravel(), gy.ravel()], 1) @ W + b)
    cls = pg.argmax(1).reshape(gx.shape)
    light = ["#E8EAF6", "#FBE9E7", "#E0F2F1"]
    ax.contourf(gx, gy, cls, levels=[-0.5, 0.5, 1.5, 2.5], colors=light)
    ax.contour(gx, gy, pg.max(1).reshape(gx.shape), levels=[0.9], colors=[C_INK], linewidths=0.8, linestyles=":")
    markers = ["o", "s", "^"]
    for k in range(3):
        m = y_tr == k
        ax.scatter(x_tr[m, 0], x_tr[m, 1], s=16, color=SERIES[k], marker=markers[k], label=CLASSES[k],
                   edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("センサ 1")
    ax.set_ylabel("センサ 2")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10, handletextpad=0.2)
    ax.text(2.9, -2.9, "点線: 確率 0.9 の境目", ha="right", va="bottom", fontsize=9.5, color=C_INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch02_regions.png")
    plt.close(fig)
    saved("ch02_regions.png")


if __name__ == "__main__":
    main()
