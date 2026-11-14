"""演習 1: データから関数を作る（回帰と過学習）

実行: python3 vla_math/ch01_regression.py
出力: docs/vla-math/figures/ch01_fit.png, ch01_degree.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

場面: 入力 x から出力 y を予測する関数を、データから作る. 本当の関係は
y = sin(2πx) だが、学習する側はそれを知らない. 測ったデータには雑音が乗る.

(1) 正規方程式で、1 次・3 次・9 次の多項式を当てはめ、学習データと検証データの誤差を比べる
(2) 同じ問題を勾配降下法で解き、正規方程式の答えに近づくことを確かめる. 学習率が大きすぎると発散する
(3) 9 次の多項式に正則化を足すと、検証データの誤差が下がる
(4) データを増やすと、9 次の多項式でも過学習が減る
"""

from __future__ import annotations

import numpy as np

from common import C_A, C_B, C_MUTED, saved, setup_matplotlib, FIG_DIR

NOISE = 0.2


def true_f(x):
    return np.sin(2 * np.pi * x)


def make_data(rng, n):
    x = np.sort(rng.uniform(0, 1, n))
    return x, true_f(x) + NOISE * rng.standard_normal(n)


def features(x, degree):
    """多項式の特徴 [1, x, x^2, ..., x^degree] を並べた行列."""
    return np.stack([x**k for k in range(degree + 1)], axis=1)


def fit_least_squares(x, y, degree, lam=0.0):
    """二乗誤差（と正則化）を最小にする重みを、正規方程式で求める."""
    phi = features(x, degree)
    a = phi.T @ phi + lam * len(x) * np.eye(degree + 1)
    return np.linalg.solve(a, phi.T @ y)


def rmse(w, x, y):
    return float(np.sqrt(np.mean((features(x, len(w) - 1) @ w - y) ** 2)))


def gradient_descent(x, y, degree, lr, steps, log_at=()):
    """平均二乗誤差 L(w) = mean((Φw - y)^2) を、勾配降下法で小さくする."""
    phi = features(x, degree)
    w = np.zeros(degree + 1)
    history = {}
    for k in range(steps + 1):
        r = phi @ w - y
        if k in log_at:
            history[k] = float(np.mean(r**2))
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e12:
            history[k] = float("inf")
            break
        w = w - lr * (2.0 / len(x)) * (phi.T @ r)    # 勾配は (2/n) Φᵀ(Φw - y)
    return w, history


def main():
    rng = np.random.default_rng(0)
    x_tr, y_tr = make_data(rng, 10)        # 学習データ
    x_va, y_va = make_data(rng, 200)       # 検証データ（学習には使わない）

    print("(1) 正規方程式で多項式を当てはめる（学習データ 10 点、検証データ 200 点）")
    print("    次数   重みの数   学習データの誤差   検証データの誤差")
    curve = []
    for d in range(10):
        w = fit_least_squares(x_tr, y_tr, d)
        tr, va = rmse(w, x_tr, y_tr), rmse(w, x_va, y_va)
        curve.append((d, tr, va))
        if d in (0, 1, 3, 6, 9):
            print(f"    {d:4d}   {d + 1:8d}   {tr:16.3f}   {va:16.3f}")
    print(f"    （誤差は二乗平均の平方根. 雑音の大きさは {NOISE}）")

    print("\n(2) 勾配降下法で 3 次の多項式を学習する")
    phi = features(x_tr, 3)
    hess = (2.0 / len(x_tr)) * phi.T @ phi
    lam_max = float(np.linalg.eigvalsh(hess).max())
    lr = 1.0 / lam_max
    w_ls = fit_least_squares(x_tr, y_tr, 3)
    w_gd, hist = gradient_descent(x_tr, y_tr, 3, lr, 200000, log_at=(0, 100, 1000, 10000, 200000))
    print(f"    学習率 {lr:.3f}（損失の曲がり具合の最大値 {lam_max:.2f} の逆数）")
    for k, v in hist.items():
        print(f"    {k:6d} 回目: 学習データの平均二乗誤差 {v:.4f}")
    print(f"    正規方程式の答えの平均二乗誤差: {rmse(w_ls, x_tr, y_tr) ** 2:.4f}")
    print(f"    正規方程式の答えとの差（重みの差の最大値）: {np.abs(w_gd - w_ls).max():.2e}")
    _, hist_big = gradient_descent(x_tr, y_tr, 3, 2.2 / lam_max, 2000, log_at=(0, 10, 100, 1000))
    vals = ", ".join(f"{k} 回目 {v:.3g}" for k, v in hist_big.items())
    print(f"    学習率を {2.2 / lam_max:.3f}（2 / 最大値 を超える）にすると: {vals}")

    print("\n(3) 9 次の多項式に正則化を足す（重みの二乗和に係数 λ を掛けて損失に足す）")
    for lam in (0.0, 1e-6, 1e-4, 1e-2):
        w = fit_least_squares(x_tr, y_tr, 9, lam)
        print(f"    λ = {lam:7.0e}: 学習 {rmse(w, x_tr, y_tr):.3f}、検証 {rmse(w, x_va, y_va):.3f}、"
              f"重みの最大の大きさ {np.abs(w).max():10.1f}")

    print("\n(4) データを増やして、9 次の多項式を当てはめる")
    for n in (10, 30, 100, 1000):
        if n == 10:
            x_n, y_n = x_tr, y_tr                  # (1) と同じデータ
        else:
            x_n, y_n = make_data(np.random.default_rng(10 + n), n)
        w = fit_least_squares(x_n, y_n, 9)
        print(f"    学習データ {n:4d} 点: 学習 {rmse(w, x_n, y_n):.3f}、検証 {rmse(w, x_va, y_va):.3f}")

    plt = setup_matplotlib()
    xs = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 7.2), sharex=True)
    for ax, d in zip(axes, (1, 3, 9)):
        w = fit_least_squares(x_tr, y_tr, d)
        ax.plot(xs, true_f(xs), color=C_MUTED, lw=1.4, ls="--", label="本当の関数")
        ax.plot(xs, features(xs, d) @ w, color=C_A, lw=2.0, label="当てはめた多項式")
        ax.scatter(x_tr, y_tr, s=34, color=C_B, zorder=3, label="学習データ")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(f"{d} 次（重み {d + 1} 個）　検証の誤差 {rmse(w, x_va, y_va):.2f}", loc="left")
        ax.set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=9.5)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch01_fit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ds = [c[0] for c in curve]
    ax.plot(ds, [c[1] for c in curve], color=C_A, marker="o", label="学習データの誤差")
    ax.plot(ds, [c[2] for c in curve], color=C_B, marker="s", label="検証データの誤差")
    ax.set_yscale("log")
    ax.set_xlabel("多項式の次数")
    ax.set_ylabel("誤差（対数目盛）")
    ax.set_xticks(ds)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch01_degree.png")
    plt.close(fig)
    saved("ch01_fit.png", "ch01_degree.png")


if __name__ == "__main__":
    main()
