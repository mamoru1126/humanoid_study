"""演習 6: 画像をトークンにする（ViT のパッチ）

実行: python3 vla_math/ch06_patches.py
出力: docs/vla-math/figures/ch06_patches.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

(1) 224×224 画素のカラー画像（机の上のブロック、コップ、ボールを描いたもの）を、16×16 画素のパッチに切る
(2) パッチを 1 列のベクトルにし、行列を掛けてトークンにする. パッチを並べ直すと、元の画像に戻る
(3) 解像度とパッチの大きさから、トークンの数と、注意の重みの数を求める
(4) 位置の埋め込みを足すと、パッチの順番を入れ替えたことが、出力に表れるようになる
"""

from __future__ import annotations

import numpy as np

from common import C_INK, saved, setup_matplotlib, FIG_DIR


def draw_scene(R=224):
    """机の上に、赤いブロック、青いコップ、緑のボールがある画像（値は 0〜1）."""
    yy, xx = np.mgrid[0:R, 0:R] / R
    img = np.zeros((R, R, 3))
    img[:] = [0.90, 0.91, 0.93]                                   # 壁
    table = yy > 0.58
    img[table] = [0.72, 0.56, 0.40]                               # 机
    img[(yy > 0.56) & (yy <= 0.58)] = [0.55, 0.40, 0.27]          # 机の縁
    block = (xx > 0.14) & (xx < 0.34) & (yy > 0.42) & (yy < 0.62)
    img[block] = [0.85, 0.18, 0.15]
    cup = (xx > 0.44) & (xx < 0.58) & (yy > 0.36) & (yy < 0.63)
    img[cup] = [0.20, 0.35, 0.80]
    ball = (xx - 0.76) ** 2 + (yy - 0.53) ** 2 < 0.085**2
    img[ball] = [0.20, 0.65, 0.30]
    return img


def patchify(img, P):
    """(R, R, 3) の画像を、(パッチの数, P*P*3) の行列にする. 並びは左上から右へ、行ごと."""
    R = img.shape[0]
    g = R // P
    x = img.reshape(g, P, g, P, 3).transpose(0, 2, 1, 3, 4)
    return x.reshape(g * g, P * P * 3)


def unpatchify(tokens, P, R):
    g = R // P
    x = tokens.reshape(g, g, P, P, 3).transpose(0, 2, 1, 3, 4)
    return x.reshape(R, R, 3)


def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(Q.shape[1])
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(axis=1, keepdims=True)
    return A @ V


def main():
    rng = np.random.default_rng(0)
    R, P, D = 224, 16, 64
    img = draw_scene(R)

    print("(1) パッチに切る")
    patches = patchify(img, P)
    print(f"    画像 {R}×{R}×3 → パッチ {patches.shape[0]} 枚、1 枚は {P}×{P}×3 = {patches.shape[1]} 個の数")

    print("\n(2) 行列 E を掛けてトークンにする（ここでは E は乱数. 本物は学習で決める）")
    E = rng.standard_normal((patches.shape[1], D)) / np.sqrt(patches.shape[1])
    tokens = patches @ E
    print(f"    トークンの列の形 {tokens.shape}（{tokens.shape[0]} 個 × {D} 次元）")
    back = unpatchify(patches, P, R)
    print(f"    パッチを並べ直した画像と元の画像の差の最大値 {np.abs(back - img).max():.1e}")

    print("\n(3) トークンの数")
    print("    解像度   パッチ   トークン   注意の重み（1 層・1 ヘッド）")
    for res, p in ((224, 16), (224, 14), (336, 14), (448, 14), (896, 14)):
        m = (res // p) ** 2
        print(f"    {res:6d}   {p:6d}   {m:8d}   {m * m:>14,}")

    print("\n(4) 位置の埋め込み")
    Wq, Wk, Wv = (rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(3))
    perm = rng.permutation(len(tokens))
    pos = 0.5 * rng.standard_normal(tokens.shape)               # 位置ごとのベクトル（本物は学習する）
    for name, add_pos in (("なし", False), ("あり", True)):
        X = tokens + (pos if add_pos else 0)
        Xp = tokens[perm] + (pos if add_pos else 0)              # 位置の埋め込みは「場所」に付くので、入れ替わらない
        Y = self_attention(X, Wq, Wk, Wv)
        Yp = self_attention(Xp, Wq, Wk, Wv)
        print(f"    位置の埋め込み{name}: 入れ替えた画像の出力 と 出力を入れ替えたもの の差の最大値 {np.abs(Yp - Y[perm]).max():.1e}")

    plt = setup_matplotlib()
    fw, fh = 4.6, 6.0
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.88 * fw / fh])
    ax.imshow(img, extent=[0, R, R, 0])
    for k in range(0, R + 1, P):
        ax.axhline(k, color="white", lw=0.8)
        ax.axvline(k, color="white", lw=0.8)
    row = 8
    ax.add_patch(plt.Rectangle((0, row * P), R, P, fill=False, edgecolor=C_INK, lw=2.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{R}×{R} 画素を {P}×{P} 画素に切ると {(R // P) ** 2} 枚", loc="left", fontsize=11)
    g = R // P
    for j in range(g):
        a2 = fig.add_axes([0.06 + j * (0.88 / g) + 0.003, 0.13, 0.88 / g - 0.006, (0.88 / g - 0.006) * fw / fh])
        a2.imshow(patches[row * g + j].reshape(P, P, 3))
        a2.set_xticks([])
        a2.set_yticks([])
        for s in a2.spines.values():
            s.set_visible(True)
            s.set_color("#9AA3B0")
        a2.grid(False)
    fig.text(0.06, 0.215, f"太枠の行の {g} 枚を、左から順に並べる（トークン {row * g + 1}〜{row * g + g} 番）", fontsize=10, color=C_INK)
    fig.text(0.06, 0.06, f"1 枚 = {P}×{P}×3 = {P * P * 3} 個の数 → 行列を掛けて 1 つのトークンにする", fontsize=10, color=C_INK)
    fig.savefig(FIG_DIR / "ch06_patches.png")
    plt.close(fig)
    saved("ch06_patches.png")


if __name__ == "__main__":
    main()
