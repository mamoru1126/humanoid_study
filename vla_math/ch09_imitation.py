"""演習 9: 模倣学習（行動クローニングと DAgger）

実行: python3 vla_math/ch09_imitation.py
出力: docs/vla-math/figures/ch09_bc_dagger.png
必要: numpy, matplotlib（GPU は不要. 数秒で終わる）

場面: 放っておくと倒れていく（ずれが広がる）1 次元の系を、まっすぐに保つ.
    x[t+1] = x[t] + Δt (λ x[t] + u[t]) + 雑音,  λ > 0 なので、u = 0 だとずれが広がる
お手本（専門家）は u = -3x で立て直す. |x| が 1 を超えたら失敗.

学習する方策は、いちばん単純な「表」にする. x を細かい区間に分け、区間ごとに、
そこで見たお手本の行動の平均を覚える. 一度も見ていない区間では、何もしない（u = 0）.

(1) 行動クローニング: お手本が走った記録だけで表を作る
(2) DAgger: 学習した方策で走らせ、訪れた状態にお手本の行動を付けて、データに足す. これを繰り返す
(3) 課題の長さ T を変えて、失敗の割合を比べる
"""

from __future__ import annotations

import numpy as np

from common import C_A, C_B, C_MUTED, saved, setup_matplotlib, FIG_DIR

LAM, DT, NOISE, LIMIT = 1.0, 0.1, 0.03, 1.0
BIN = 0.02
EDGES = np.arange(-LIMIT, LIMIT + BIN, BIN)


def expert(x):
    return -3.0 * x


class TablePolicy:
    def __init__(self):
        self.sum = np.zeros(len(EDGES) + 1)
        self.cnt = np.zeros(len(EDGES) + 1)

    def add(self, xs, us):
        k = np.digitize(xs, EDGES)
        np.add.at(self.sum, k, us)
        np.add.at(self.cnt, k, 1)

    def __call__(self, x):
        k = np.digitize(x, EDGES)
        return np.where(self.cnt[k] > 0, self.sum[k] / np.maximum(self.cnt[k], 1), 0.0)

    def covered(self):
        k = np.nonzero(self.cnt[1:-1])[0]
        return EDGES[k.min()], EDGES[k.max() + 1]


def rollout(policy, rng, T, n, x0_std=0.05):
    """n 本を同時に走らせる. 返り値: 状態の記録 (n, T+1) と、失敗したか."""
    x = x0_std * rng.standard_normal(n)
    xs = np.full((n, T + 1), np.nan)
    xs[:, 0] = x
    alive = np.ones(n, dtype=bool)
    for t in range(T):
        u = policy(x)
        x = np.where(alive, x + DT * (LAM * x + u) + NOISE * rng.standard_normal(n), x)
        xs[:, t + 1] = np.where(alive, x, np.nan)      # 失敗した瞬間までを記録する
        alive &= np.abs(x) <= LIMIT
    return xs, ~alive


def visited_states(xs):
    v = xs[np.isfinite(xs)]
    return v[np.abs(v) <= LIMIT]


def evaluate(policy, T, n=2000, seed=100):
    """失敗の割合と、コストの平均. コスト = 失敗してから課題の終わりまでのステップ数（失敗しなければ 0）."""
    xs, failed = rollout(policy, np.random.default_rng(seed + T), T, n)
    t_fail = np.where(failed, np.argmax(np.abs(np.nan_to_num(xs, nan=0.0)) > LIMIT, axis=1), T)
    return float(failed.mean()), float(np.mean(T - t_fail))


def main():
    rng = np.random.default_rng(0)
    print("(1) 行動クローニング: お手本の記録（100 ステップ × 5 本）から表を作る")
    demo, _ = rollout(expert, rng, 100, 5)
    bc = TablePolicy()
    s = visited_states(demo)
    bc.add(s, expert(s))
    lo, hi = bc.covered()
    print(f"    お手本が訪れた範囲: {lo:+.2f} 〜 {hi:+.2f}（この外では、方策は何もしない）")

    print("\n(2) DAgger: 方策で走らせ、訪れた状態にお手本の行動を付けて足す（1 回に 100 ステップ × 5 本）")
    dagger = TablePolicy()
    dagger.add(s, expert(s))
    for it in range(1, 6):
        xs, failed = rollout(dagger, rng, 100, 5)
        v = visited_states(xs)
        dagger.add(v, expert(v))
        lo, hi = dagger.covered()
        print(f"    {it} 回目: 走らせた 5 本のうち失敗 {int(failed.sum())} 本、データのある範囲 {lo:+.2f} 〜 {hi:+.2f}")

    print("\n(3) 課題の長さ T と、失敗の割合・コスト（各 2000 本）")
    print("    コスト = 失敗してから課題の終わりまでのステップ数の平均")
    print("        T   お手本   行動クローニング（失敗 / コスト / コスト÷T）   DAgger（失敗 / コスト）")
    for T in (50, 100, 200, 400, 800):
        e, _ = evaluate(expert, T)
        b, cb = evaluate(bc, T)
        d, cd = evaluate(dagger, T)
        print(f"    {T:5d}   {100 * e:5.1f}%   {100 * b:5.1f}% / {cb:6.1f} / {cb / T:.3f}"
              f"                   {100 * d:5.1f}% / {cd:4.1f}")
    total = sum(len(visited_states(x)) for x in [demo])
    print(f"    （行動クローニングのデータ {total} 点、DAgger のデータ {int(dagger.cnt.sum())} 点）")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.6), sharex=True)
    T = 400
    for ax, pol, title in ((axes[0], bc, "行動クローニング"), (axes[1], dagger, "DAgger（5 回繰り返したあと）")):
        lo, hi = pol.covered()
        ax.axhspan(lo, hi, color="#E8EAF6", zorder=0)
        xs, failed = rollout(pol, np.random.default_rng(7), T, 30)
        for i in range(len(xs)):
            ax.plot(xs[i], color=C_B if failed[i] else C_A, lw=1.4 if failed[i] else 0.8,
                    alpha=1.0 if failed[i] else 0.55)
        ax.axhline(LIMIT, color=C_MUTED, ls="--", lw=1)
        ax.axhline(-LIMIT, color=C_MUTED, ls="--", lw=1)
        ax.set_ylim(-1.25, 1.25)
        ax.set_ylabel("ずれ x")
        ax.set_title(f"{title}　30 本中 失敗 {int(failed.sum())} 本", loc="left")
    axes[0].text(T, 1.08, "失敗の線（|x| = 1）", ha="right", va="bottom", fontsize=9.5, color=C_MUTED)
    axes[1].set_xlabel("ステップ")
    axes[1].text(T, -1.2, "薄い帯: 学習データのある範囲", ha="right", va="bottom", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch09_bc_dagger.png")
    plt.close(fig)
    saved("ch09_bc_dagger.png")


if __name__ == "__main__":
    main()
