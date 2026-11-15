"""演習 3.2: ニューロンとパーセプトロン

実行: python3 vla_math/ch03_2_perceptron.py
出力: docs/vla-math/figures/ch03_2_gates.png, ch03_2_learning.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

(1) しきい値で 0 か 1 を出すニューロンに、重みを手で入れて、AND、OR、NAND を作る
(2) パーセプトロンの学習則で、2 つの集まりを分ける直線を学習する. 誤りの回数を、ノビコフの上限と比べる
(3) 同じ学習則を XOR に使うと、いつまでも誤りが 0 にならない
(4) しきい値の代わりにシグモイドを使っても（ロジスティック回帰）、XOR は分けられない
"""

from __future__ import annotations

import numpy as np

from common import C_A, C_B, C_INK, C_MUTED, saved, setup_matplotlib, FIG_DIR

X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR": np.array([0, 1, 1, 0]),
}


def step(z):
    return (z >= 0).astype(int)


def neuron(x, w, b):
    """形式ニューロン: 重み付きの和がしきい値を超えたら 1."""
    return step(x @ w + b)


def make_separable(n, gap, rng):
    """本当の境目 x1 + 2 x2 = 0.5 から gap 以上離れた点だけを n 個集める（直線で分けられるデータ）."""
    out = np.empty((0, 2))
    while len(out) < n:
        pts = rng.uniform(-2, 2, (4 * n, 2))
        dist = (pts @ np.array([1.0, 2.0]) - 0.5) / np.sqrt(5)
        out = np.vstack([out, pts[np.abs(dist) > gap]])
    X = out[:n]
    t = np.where(X @ np.array([1.0, 2.0]) - 0.5 > 0, 1.0, -1.0)
    return X, t


def perceptron(X, t, epochs, rng=None, snapshots=()):
    """パーセプトロンの学習則. t は +1 か -1. 誤ったときだけ w ← w + t x（x には 1 を足して、b も w に含める）.

    点を 1 つずつ順に見て、誤った点で直す. 次の誤りを探す部分だけを numpy でまとめて計算している
    （1 点ずつ見るのと結果は同じ）.
    """
    Xa = np.hstack([X, np.ones((len(X), 1))])
    n = len(X)
    w = np.zeros(Xa.shape[1])
    mistakes_per_epoch, snaps, total = [], {}, 0
    for _ in range(epochs):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        Xo, to = Xa[order], t[order]
        pos, m = 0, 0
        while pos < n:
            bad = np.flatnonzero(to[pos:] * (Xo[pos:] @ w) <= 0)   # 誤り（境目の上も誤りとする）
            if len(bad) == 0:
                break
            i = pos + bad[0]
            w = w + to[i] * Xo[i]
            m += 1
            total += 1
            if total in snapshots:
                snaps[total] = w.copy()
            pos = i + 1
        mistakes_per_epoch.append(m)
        if m == 0:
            break
    return w, mistakes_per_epoch, total, snaps


def logistic_regression(X, t01, steps=5000, lr=0.5):
    """しきい値をシグモイドに替え、交差エントロピーを勾配降下法で小さくする."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(3)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xa @ w)))
        w -= lr * Xa.T @ (p - t01) / len(X)       # 勾配は (予測 - 正解) × 入力
    p = 1 / (1 + np.exp(-(Xa @ w)))
    return w, p


def main():
    print("(1) 重みを手で入れて論理回路を作る（出力 = 1[w1 x1 + w2 x2 + b ≥ 0]）")
    hand = {"AND": ((1, 1), -1.5), "OR": ((1, 1), -0.5), "NAND": ((-1, -1), 1.5)}
    for name, (w, b) in hand.items():
        y = neuron(X4, np.array(w, float), b)
        ok = "一致" if np.array_equal(y, TARGETS[name]) else "不一致"
        print(f"    {name:4s}: w = {w}, b = {b:+.1f} → 出力 {y.tolist()}（正解 {TARGETS[name].tolist()}、{ok}）")

    print("\n(2) パーセプトロンの学習則で、2 つの集まりを分ける")
    Xp, tp = make_separable(80, 0.1, np.random.default_rng(0))
    w_sep, mpe, total, snaps = perceptron(Xp, tp, 100, rng=np.random.default_rng(1), snapshots=(1, 2, 5))
    Xa = np.hstack([Xp, np.ones((len(Xp), 1))])
    R = np.max(np.linalg.norm(Xa, axis=1))
    u = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])   # 本当の境目（入力に 1 を足した 3 次元で）
    gamma = np.min(tp * (Xa @ u))
    print(f"    点 80 個. 1 周ごとの誤りの回数: {mpe}")
    print(f"    誤りの合計 {total} 回で収束した（最後の 1 周で誤り 0）")
    print(f"    ノビコフの上限 (R/γ)²: R = {R:.2f}, γ = {gamma:.4f}（本当の境目で測った余白）→ {(R / gamma) ** 2:.0f} 回")
    print("    すき間を狭めると、誤りの回数が増える（点 1000 個. データと並べ方を変えて 100 回の平均）")
    for gap in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        tot, bounds = [], []
        for s_ in range(100):
            X_, t_ = make_separable(1000, gap, np.random.default_rng(100 + s_))
            Xa_ = np.hstack([X_, np.ones((len(X_), 1))])
            bounds.append((np.max(np.linalg.norm(Xa_, axis=1)) / np.min(t_ * (Xa_ @ u))) ** 2)
            tot.append(perceptron(X_, t_, 100000, rng=np.random.default_rng(s_))[2])
        print(f"      すき間 {gap:4.2f}: 誤り 平均 {np.mean(tot):6.1f} 回（最大 {max(tot):4d}）、上限 (R/γ)² の平均 {np.mean(bounds):8.0f}")

    print("\n(3) XOR に同じ学習則を使う（100 周）")
    t_xor = np.where(TARGETS["XOR"] == 1, 1.0, -1.0)
    _, mpe_x, total_x, _ = perceptron(X4, t_xor, 100)
    print(f"    1 周ごとの誤りの回数（最初の 10 周）: {mpe_x[:10]}")
    print(f"    100 周たっても誤りは 0 にならない（最後の周の誤り {mpe_x[-1]} 回、合計 {total_x} 回）")
    best = 0
    for w1 in np.linspace(-2, 2, 41):
        for w2 in np.linspace(-2, 2, 41):
            for b in np.linspace(-2, 2, 41):
                best = max(best, np.mean(neuron(X4, np.array([w1, w2]), b) == TARGETS["XOR"]))
    print(f"    重みを細かく総当たりしても、4 点のうち正解できるのは最大 {int(best * 4)} 点")

    print("\n(4) シグモイドと交差エントロピー（ロジスティック回帰）で学習する")
    for name in ("AND", "XOR"):
        w, p = logistic_regression(X4, TARGETS[name].astype(float))
        acc = np.mean((p >= 0.5) == TARGETS[name])
        print(f"    {name}: 出力の確率 {np.round(p, 2).tolist()}、正解率 {100 * acc:.0f}%")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.25))
    xs = np.linspace(-0.4, 1.4, 10)
    for ax, name in zip(axes, ("AND", "OR", "XOR")):
        t = TARGETS[name]
        for k, (c, mk, lab) in enumerate(((C_B, "o", "0"), (C_A, "s", "1"))):
            m = t == k
            ax.scatter(X4[m, 0], X4[m, 1], s=110, color=c, marker=mk, zorder=3, label=f"出力 {lab}",
                       edgecolors="white", linewidths=1.0)
        if name in hand:
            (w1, w2), b = hand[name]
            ax.plot(xs, -(w1 * xs + b) / w2, color=C_INK, lw=1.6)
        else:
            ax.plot(xs, 0.5 - xs, color=C_MUTED, lw=1.3, ls="--")
            ax.plot(xs, 1.5 - xs, color=C_MUTED, lw=1.3, ls="--")
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.45, 1.4)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("x1")
        ax.set_aspect("equal")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("x2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIG_DIR / "ch03_2_gates.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.7), sharex=True, sharey=True)
    xs = np.linspace(-2.5, 2.8, 10)
    stages = [(1, snaps.get(1)), (2, snaps.get(2)), (5, snaps.get(5)), (total, w_sep)]
    for ax, (k, wk) in zip(axes.ravel(), stages):
        ax.plot(xs, (0.5 - xs) / 2, color=C_MUTED, lw=1.0, ls="--")     # 本当の境目
        ax.scatter(Xp[tp > 0, 0], Xp[tp > 0, 1], s=10, color=C_A, marker="s", linewidths=0)
        ax.scatter(Xp[tp < 0, 0], Xp[tp < 0, 1], s=10, color=C_B, marker="o", linewidths=0)
        if wk is not None and abs(wk[1]) > 1e-9:
            ax.plot(xs, -(wk[0] * xs + wk[2]) / wk[1], color=C_INK, lw=1.6)
        title = f"{k} 回目の修正のあと" if k != total else f"収束（修正 {total} 回）"
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlim(-2.5, 2.8)
        ax.set_ylim(-2.3, 2.8)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch03_2_learning.png")
    plt.close(fig)
    saved("ch03_2_gates.png", "ch03_2_learning.png")


if __name__ == "__main__":
    main()
