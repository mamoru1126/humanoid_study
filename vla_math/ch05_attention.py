"""演習 5: 注意機構と Transformer

実行: python3 vla_math/ch05_attention.py
出力: docs/vla-math/figures/ch05_attention.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

学習はしない. 注意機構の性質を、計算して確かめる.
(1) 手で作った小さな例で、各トークンがどのトークンを見るかを、注意の重みとして見る
(2) 因果マスク: 先のトークンを入れ替えても、それより前の出力は変わらない
(3) 位置の情報がないと、トークンの順番を入れ替えても、出力は同じように入れ替わるだけ
(4) KV キャッシュ: 1 つずつ出すとき、過去の K と V を取っておけば、全部を計算し直した結果と一致する
(5) Transformer の重みの数と、注意の計算量
"""

from __future__ import annotations

import numpy as np

from common import C_INK, blue_map, saved, setup_matplotlib, FIG_DIR


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, causal=False, bias=None):
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d + マスク) V. 重みも返す."""
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    if bias is not None:
        S = S + bias
    if causal:
        n = S.shape[0]
        S = np.where(np.tril(np.ones((n, n), dtype=bool)), S, -np.inf)
    A = softmax(S)
    return A @ V, A


def multi_head(X, Wq, Wk, Wv, Wo, n_heads, causal):
    n, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs = []
    for h in range(n_heads):
        s = slice(h * dh, (h + 1) * dh)
        o, _ = attention(Q[:, s], K[:, s], V[:, s], causal=causal)
        outs.append(o)
    return np.concatenate(outs, axis=1) @ Wo


def toy_example():
    """手で作った例. 特徴は 5 種類: 文の始め、色、物、助詞、動詞."""
    tokens = ["<始>", "赤い", "ブロック", "を", "青い", "箱", "に", "入れて"]
    kinds = [0, 1, 2, 3, 1, 2, 3, 4]
    X = np.eye(5)[kinds]                       # トークンの特徴（one-hot）
    # 問い合わせ（Q）: 物は色を探し、動詞は物を探す. 助詞と色は文の始めを見る（何もしない）
    Wq = np.zeros((5, 5))
    Wq[0, 0] = 4.0
    Wq[1, 0] = 4.0
    Wq[2, 1] = 8.0
    Wq[3, 0] = 4.0
    Wq[4, 2] = 8.0
    Wk = np.eye(5)
    n = len(tokens)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    bias = -0.6 * dist                          # 近いトークンを少し優先する（位置の情報）
    _, A = attention(X @ Wq * np.sqrt(5), X @ Wk, X, causal=True, bias=bias)
    _, A_nopos = attention(X @ Wq * np.sqrt(5), X @ Wk, X, causal=True)
    return tokens, A, A_nopos


def main():
    rng = np.random.default_rng(0)

    print("(1) 手で作った例: 各行のトークンが、どのトークンをどれだけ見るか")
    tokens, A, A_nopos = toy_example()
    for i in (2, 5, 7):
        top = np.argsort(-A[i])[:2]
        print(f"    「{tokens[i]}」→ " + "、".join(f"「{tokens[j]}」{A[i, j]:.2f}" for j in top)
              + f"（行の合計 {A[i].sum():.2f}）")
    print(f"    位置の情報を外すと、「箱」が見る重み: 赤い {A_nopos[5, 1]:.2f}、青い {A_nopos[5, 4]:.2f}"
          "（どちらの色か決められない）")

    n, d, h = 10, 32, 4
    X = rng.standard_normal((n, d))
    Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(4)]

    print("\n(2) 因果マスク: 7 番目より後のトークンを入れ替える")
    X2 = X.copy()
    X2[7:] = rng.standard_normal((n - 7, d))
    for causal in (True, False):
        Y1 = multi_head(X, *Ws, h, causal)
        Y2 = multi_head(X2, *Ws, h, causal)
        diff = np.abs(Y1[:7] - Y2[:7]).max()
        print(f"    マスク{'あり' if causal else 'なし'}: 0〜6 番目の出力の変化の最大値 {diff:.1e}")

    print("\n(3) 位置の情報がないとき、入力の順番を入れ替える")
    perm = rng.permutation(n)
    Y = multi_head(X, *Ws, h, causal=False)
    Yp = multi_head(X[perm], *Ws, h, causal=False)
    print(f"    入れ替えた入力の出力 と 出力を同じく入れ替えたもの の差の最大値 {np.abs(Yp - Y[perm]).max():.1e}")
    print("    （注意機構だけでは、順番を区別できない. だから位置の情報を足す）")

    print("\n(4) KV キャッシュ（1 つずつ出す場合）")
    Wq, Wk, Wv, Wo = Ws
    full = multi_head(X, Wq, Wk, Wv, Wo, h, causal=True)
    K_cache, V_cache, outs = [], [], []
    dh = d // h
    for t in range(n):
        x = X[t:t + 1]
        K_cache.append(x @ Wk)
        V_cache.append(x @ Wv)
        K = np.concatenate(K_cache)
        V = np.concatenate(V_cache)
        q = x @ Wq
        o = []
        for hh in range(h):
            s = slice(hh * dh, (hh + 1) * dh)
            oh, _ = attention(q[:, s], K[:, s], V[:, s])
            o.append(oh)
        outs.append(np.concatenate(o, axis=1) @ Wo)
    print(f"    全部を一度に計算した出力との差の最大値 {np.abs(np.concatenate(outs) - full).max():.1e}")
    for N in (100, 1000):
        no_cache = sum(t * t for t in range(1, N + 1))
        cache = sum(range(1, N + 1))
        print(f"    {N} 個を 1 つずつ出すときの、内積の数: キャッシュなし {no_cache:.2e}、あり {cache:.2e}（{no_cache / cache:.0f} 倍）")

    print("\n(5) 重みの数と、注意の計算量")
    L, dm, vocab, ctx = 12, 768, 50257, 1024
    per_layer = 12 * dm * dm + 13 * dm
    total = vocab * dm + ctx * dm + L * per_layer + 2 * dm
    print(f"    1 層あたり 12d² + 13d = {per_layer:,} 個（d = {dm}）")
    print(f"    12 層、語彙 {vocab}、位置 {ctx} 個のとき、全体で {total:,} 個")
    for m in (196, 256, 1024, 4096):
        print(f"    トークン {m:5d} 個: 注意の重みは 1 層・1 ヘッドで {m * m:>12,} 個")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    shown = np.where(np.tril(np.ones_like(A, dtype=bool)), A, np.nan)
    im = ax.imshow(shown, cmap=blue_map(), vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=10)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("見られるトークン（キー）")
    ax.set_ylabel("見るトークン（クエリ）")
    ax.grid(False)
    for i in range(len(tokens)):
        for j in range(i + 1):
            if A[i, j] >= 0.15:
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if A[i, j] > 0.55 else C_INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("注意の重み")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch05_attention.png")
    plt.close(fig)
    saved("ch05_attention.png")


if __name__ == "__main__":
    main()
