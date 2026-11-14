"""演習 7: 画像と言葉を結ぶ（CLIP と SigLIP の対照学習）

実行: python3 vla_math/ch07_contrastive.py
出力: docs/vla-math/figures/ch07_similarity.png
必要: numpy, matplotlib（GPU は不要. 10 秒ほど）

本物の画像と文の代わりに、作り物の「画像の特徴」と「文の特徴」を使う.
どちらも、物の色（赤・青・緑）と種類（ブロック・コップ・ボール）から作る. 画像には位置や明るさの違い、
文には言い回しの違いが混ざる. 「緑のボール」の組だけは、学習データに入れない.

(1) 対照学習の損失（CLIP の softmax 型、SigLIP のシグモイド型）の勾配を、数値微分で確かめる
(2) 2 つのエンコーダ（ここでは行列 1 つずつ）を学習する
(3) 学習のあと、画像から文を探す. 文の候補だけを並べて分類する（ゼロショット分類）.
    学習で一度も見ていない「緑のボール」も当てられるかを見る
"""

from __future__ import annotations

import numpy as np

from common import C_INK, blue_map, saved, setup_matplotlib, FIG_DIR

COLORS = ["赤い", "青い", "緑の"]
OBJECTS = ["ブロック", "コップ", "ボール"]
CONCEPTS = [(c, o) for c in range(3) for o in range(3)]
HELD_OUT = (2, 2)                      # 緑のボール: 学習に使わない
DX, DT, DE = 32, 24, 8                  # 画像の特徴、文の特徴、共通の空間の次元


class World:
    """画像の特徴と文の特徴を作る、固定した仕組み（学習する側は知らない）."""

    def __init__(self, rng):
        self.A = rng.standard_normal((DX, 3 + 3 + 4)) * 0.8   # 色、種類、見え方の違い（4 次元）
        self.B = rng.standard_normal((DT, 3 + 3 + 3)) * 0.8   # 色、種類、言い回し（3 通り）

    def image(self, rng, c, o):
        z = np.concatenate([np.eye(3)[c], np.eye(3)[o], rng.standard_normal(4)])
        return np.tanh(self.A @ z) + 0.1 * rng.standard_normal(DX)

    def text(self, rng, c, o, template=None):
        k = rng.integers(3) if template is None else template
        z = np.concatenate([np.eye(3)[c], np.eye(3)[o], np.eye(3)[k]])
        return self.B @ z + (0.05 * rng.standard_normal(DT) if template is None else 0.0)

    def pairs(self, rng, n, allow_held_out):
        X, T, lab = [], [], []
        while len(X) < n:
            c, o = rng.integers(3), rng.integers(3)
            if (c, o) == HELD_OUT and not allow_held_out:
                continue
            X.append(self.image(rng, c, o))
            T.append(self.text(rng, c, o))
            lab.append(c * 3 + o)
        return np.array(X), np.array(T), np.array(lab)


def normalize(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / n, n


def norm_backward(dU, U, n):
    return (dU - U * np.sum(dU * U, axis=1, keepdims=True)) / n


def softmax(z, axis):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def clip_loss(params, X, T, scale=10.0):
    """CLIP: 行ごと（画像→文）と列ごと（文→画像）の交差エントロピーの平均."""
    Wx, Wt = params["Wx"], params["Wt"]
    U, nu = normalize(X @ Wx)
    V, nv = normalize(T @ Wt)
    L = scale * U @ V.T
    n = len(X)
    pr, pc = softmax(L, 1), softmax(L, 0)
    loss = -0.5 * (np.mean(np.log(np.diag(pr))) + np.mean(np.log(np.diag(pc))))
    G = 0.5 * ((pr - np.eye(n)) + (pc - np.eye(n))) / n
    dU, dV = scale * G @ V, scale * G.T @ U
    grads = {"Wx": X.T @ norm_backward(dU, U, nu), "Wt": T.T @ norm_backward(dV, V, nv)}
    return loss, grads


def siglip_loss(params, X, T, scale=10.0):
    """SigLIP: 全部の組について「合っている（+1）か、いないか（-1）」を当てる 2 値分類."""
    Wx, Wt, b = params["Wx"], params["Wt"], params["b"]
    U, nu = normalize(X @ Wx)
    V, nv = normalize(T @ Wt)
    L = scale * U @ V.T + b
    n = len(X)
    Z = 2 * np.eye(n) - 1
    loss = np.sum(np.logaddexp(0.0, -Z * L)) / n
    G = -Z / (1.0 + np.exp(Z * L)) / n               # d loss / d L
    dU, dV = scale * G @ V, scale * G.T @ U
    grads = {"Wx": X.T @ norm_backward(dU, U, nu), "Wt": T.T @ norm_backward(dV, V, nv), "b": np.array(G.sum())}
    return loss, grads


def grad_check(loss_fn, params, X, T):
    _, g = loss_fn(params, X, T)
    worst = 0.0
    rng = np.random.default_rng(3)
    h = 1e-6
    for key, p in params.items():
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in p.shape) if p.ndim else ()
            old = p[idx].copy() if p.ndim else p.copy()
            p[idx] = old + h
            lp = loss_fn(params, X, T)[0]
            p[idx] = old - h
            lm = loss_fn(params, X, T)[0]
            p[idx] = old
            num = (lp - lm) / (2 * h)
            ana = g[key][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + 1e-8)


def prompts(world):
    """各概念について、言い回し 0 番の文の特徴（「a photo of a ___」にあたる）."""
    rng = np.random.default_rng(0)
    return np.array([world.text(rng, c, o, template=0) for c, o in CONCEPTS])


def evaluate(params, world, X, T, lab):
    U, _ = normalize(X @ params["Wx"])
    V, _ = normalize(T @ params["Wt"])
    P, _ = normalize(prompts(world) @ params["Wt"])
    best_text = np.argmax(U @ V.T, axis=1)
    retrieval = np.mean(lab[best_text] == lab)                  # 選んだ文の概念が合っている割合
    pred = np.argmax(U @ P.T, axis=1)
    zero_shot = np.mean(pred == lab)
    held = lab == HELD_OUT[0] * 3 + HELD_OUT[1]
    return retrieval, zero_shot, np.mean(pred[held] == lab[held]), U, P


def init_params(rng, with_bias):
    p = {"Wx": rng.standard_normal((DX, DE)) / np.sqrt(DX), "Wt": rng.standard_normal((DT, DE)) / np.sqrt(DT)}
    if with_bias:
        p["b"] = np.array(-5.0)
    return p


def main():
    rng = np.random.default_rng(0)
    world = World(rng)
    X_tr, T_tr, _ = world.pairs(rng, 4000, allow_held_out=False)
    X_te, T_te, lab_te = world.pairs(rng, 900, allow_held_out=True)

    print("(1) 勾配の確認（数値微分との相対誤差の最大値）")
    Xb, Tb = X_tr[:16], T_tr[:16]
    print(f"    CLIP の損失: {grad_check(clip_loss, init_params(np.random.default_rng(1), False), Xb, Tb):.1e}")
    print(f"    SigLIP の損失: {grad_check(siglip_loss, init_params(np.random.default_rng(1), True), Xb, Tb):.1e}")

    results = {}
    for name, fn, with_bias in (("CLIP", clip_loss, False), ("SigLIP", siglip_loss, True)):
        params = init_params(np.random.default_rng(1), with_bias)
        r0, z0, h0, U0, P0 = evaluate(params, world, X_te, T_te, lab_te)
        opt = Adam(params, 0.01)
        r_b = np.random.default_rng(2)
        print(f"\n(2) {name} で学習する（1 回に 64 組、2000 回）")
        for step in range(2001):
            idx = r_b.choice(len(X_tr), 64, replace=False)
            loss, grads = fn(params, X_tr[idx], T_tr[idx])
            if step in (0, 200, 2000):
                print(f"    {step:4d} 回目: 損失 {loss:.3f}")
            if step < 2000:
                opt.step(params, grads)
        r1, z1, h1, U1, P1 = evaluate(params, world, X_te, T_te, lab_te)
        print(f"(3) {name} の結果（見ていない 900 組で測る. 当て推量なら 11%）")
        print(f"    画像から文を探して、概念が合う割合: 学習前 {100 * r0:.0f}% → 学習後 {100 * r1:.0f}%")
        print(f"    ゼロショット分類（文の候補 9 つから選ぶ）: 学習前 {100 * z0:.0f}% → 学習後 {100 * z1:.0f}%")
        print(f"    そのうち、学習で見ていない「緑のボール」: 学習前 {100 * h0:.0f}% → 学習後 {100 * h1:.0f}%")
        if with_bias:
            print(f"    学習したずらし b = {float(params['b']):.2f}")
        results[name] = (U0, P0, U1, P1)

    plt = setup_matplotlib()
    U0, P0, U1, P1 = results["CLIP"]
    names = [f"{COLORS[c]}{OBJECTS[o]}" for c, o in CONCEPTS]
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 8.4))
    for ax, U, P, title in ((axes[0], U0, P0, "学習前"), (axes[1], U1, P1, "学習後（CLIP の損失）")):
        M = np.array([U[lab_te == k].mean(0) for k in range(9)]) @ P.T
        im = ax.imshow(M, cmap=blue_map(), vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(9))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=9.5)
        ax.set_yticks(range(9))
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlabel("文の候補")
        ax.set_ylabel("画像（概念ごとの平均）")
        ax.set_title(title, loc="left")
        ax.grid(False)
        k = HELD_OUT[0] * 3 + HELD_OUT[1]
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="#C2570C", lw=2.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("コサイン類似度")
    fig.text(0.02, 0.005, "橙の枠: 学習で一度も見ていない組（緑のボール）", fontsize=9.5, color=C_INK)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG_DIR / "ch07_similarity.png")
    plt.close(fig)
    saved("ch07_similarity.png")


if __name__ == "__main__":
    main()
