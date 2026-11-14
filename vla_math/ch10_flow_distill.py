"""演習 10: 拡散とフローマッチング、そして蒸留（学習・推論・蒸留を 1 次元の問題で確かめる）

実行: python3 vla_math/ch10_flow_distill.py
出力: docs/vla-math/figures/ch10_paths.png, ch10_samples.png, ch10_map.png
必要: numpy, matplotlib（GPU は不要. CPU で 1 分以内）

場面: ロボットの正面に障害物がある. お手本は、左へ避けるもの（行動 a = -1）と、
右へ避けるもの（a = +1）が半々. 行動は 1 次元で、お手本の分布は山を 2 つ持つ.
正面（a = 0）は、お手本に 1 つもない「ぶつかる行動」である.

(1) 回帰   二乗誤差で「1 つの答え」を学習すると、2 つの山の平均（a = 0）になる
(2) 学習   フローマッチングで速度場 v(a, τ) を学習し、理論の式と比べる
(3) 推論   ノイズから始めて、オイラー法で N ステップ積分する. N を変えて比べる
(4) 蒸留   10 ステップの教師が決める写像（ノイズ → 行動）を、1 回の計算で出す生徒に写す
(5) おまけ 知識蒸留の「温度」を、手で計算できる例で見る

ニューラルネットは、逆伝播と Adam まで numpy で書いてある（MLP クラス）.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "vla-math" / "figures"

# 図の色. 2 系列までは、白地で見分けやすいことを確かめた 2 色を使う.
C_INK = "#18222D"
C_MUTED = "#8A94A0"
C_A = "#3949AB"   # 青. 左の山へ行くもの、教師
C_B = "#C2570C"   # 橙. 右の山へ行くもの、生徒

MODES = np.array([-1.0, 1.0])   # お手本の山の中心（左へ避ける、右へ避ける）
SIGMA = 0.1                     # お手本のばらつき
NEAR = 0.3                      # 山の中心からこの距離までを「山に入った」と数える


# ---------------------------------------------------------------------------
# お手本のデータと、理論の式
# ---------------------------------------------------------------------------
def sample_actions(rng, n):
    """お手本の行動. 山を同じ割合で選び、そのまわりに少しばらつかせる."""
    s = rng.integers(0, len(MODES), size=n)
    return MODES[s] + SIGMA * rng.standard_normal(n)


def exact_velocity(x, tau):
    """周辺の速度場 v*(x, τ) = E[a - ε | a^τ = x] の理論の式.

    a^τ = (1-τ) ε + τ a、ε ~ N(0,1)、a は 2 つの正規分布の混合.
    山 s について、(a, ε, a^τ) は同時に正規分布なので、条件付き期待値が式で書ける.
    """
    x = np.asarray(x, dtype=float)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), x.shape)
    var = (1 - tau) ** 2 + (tau * SIGMA) ** 2            # 山 s のもとでの a^τ の分散
    coef = (tau * SIGMA**2 - (1 - tau)) / var
    d = x[..., None] - tau[..., None] * MODES               # 各山の平均とのずれ
    logw = -0.5 * d**2 / var[..., None]
    w = np.exp(logw - logw.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)                      # x がどちらの山から来たかの確率
    return np.sum(w * (MODES + coef[..., None] * d), axis=-1)


# ---------------------------------------------------------------------------
# ニューラルネット（全結合）. 順伝播、逆伝播、Adam
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((m, n)) * np.sqrt(1.0 / m) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]   # Adam の 1 次の移動平均
        self.v = [np.zeros_like(p) for p in self.W + self.b]   # Adam の 2 次の移動平均
        self.t = 0

    def forward(self, x):
        """順伝播. 途中の値を逆伝播のために取っておく."""
        self.h = [x]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.h[-1] @ W + b
            self.h.append(z if i == last else np.tanh(z))
        return self.h[-1]

    def __call__(self, x):
        return self.forward(x)

    def train_step(self, x, target, lr):
        """二乗誤差 L = mean((f(x) - target)^2) を、勾配で 1 回だけ小さくする."""
        y = self.forward(x)
        diff = y - target
        loss = float(np.mean(diff**2))
        g = 2.0 * diff / diff.size                    # dL/dy
        grads_W, grads_b = [], []
        for i in reversed(range(len(self.W))):        # 連鎖律で、出力側から入力側へ
            grads_W.append(self.h[i].T @ g)
            grads_b.append(g.sum(axis=0))
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - self.h[i] ** 2)   # tanh の微分は 1 - tanh^2
        grads = grads_W[::-1] + grads_b[::-1]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, gp, m, v in zip(self.W + self.b, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * gp
            v *= b2
            v += (1 - b2) * gp**2
            p -= lr * (m / (1 - b1**self.t)) / (np.sqrt(v / (1 - b2**self.t)) + eps)
        return loss


def cosine_lr(step, total, lr_max, lr_min=1e-4):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / total))


# ---------------------------------------------------------------------------
# (2) 学習: フローマッチング
# ---------------------------------------------------------------------------
def train_flow(rng, steps=6000, batch=512):
    """速度場 v_θ(a^τ, τ) を学習する.

    1. お手本 a と、ノイズ ε と、時間 τ を取り出す
    2. 途中の点 a^τ = (1-τ) ε + τ a を作る
    3. その点での目標の速度 a - ε に、ネットの出力を近づける（二乗誤差）
    微分方程式は、学習では一度も解かない.
    """
    net = MLP([2, 64, 64, 1], rng)
    log = []
    for k in range(steps):
        a = sample_actions(rng, batch)
        eps = rng.standard_normal(batch)
        tau = rng.random(batch)
        x = (1 - tau) * eps + tau * a
        loss = net.train_step(np.stack([x, tau], axis=1), (a - eps)[:, None], cosine_lr(k, steps, 3e-3))
        log.append(loss)
    return net, np.array(log)


def velocity(net, x, tau):
    tau = np.broadcast_to(np.asarray(tau, dtype=float), np.shape(x))
    return net(np.stack([np.asarray(x, dtype=float), tau], axis=1))[:, 0]


# ---------------------------------------------------------------------------
# (3) 推論: 常微分方程式 da/dτ = v(a, τ) を、オイラー法で解く
# ---------------------------------------------------------------------------
def integrate(vel, eps, n_steps, keep_path=False):
    """ノイズ ε（τ = 0）から始めて、τ = 1 まで n_steps 回で進む."""
    a = np.array(eps, dtype=float)
    path = [a.copy()]
    for k in range(n_steps):
        a = a + vel(a, k / n_steps) / n_steps
        if keep_path:
            path.append(a.copy())
    return (a, np.array(path)) if keep_path else a


def in_mode(a):
    """どちらかの山に入った割合."""
    return float(np.mean(np.min(np.abs(a[:, None] - MODES), axis=1) < NEAR))


# ---------------------------------------------------------------------------
# (4) 蒸留: 教師の写像 ε → a を、1 回の計算で出す生徒に写す
# ---------------------------------------------------------------------------
def distill(rng, teacher, steps=3000, batch=512):
    """生徒 g_φ(ε) を、教師の出力を目標にした二乗誤差で学習する.

    教師: 学習済みの速度場を、オイラー法で 10 ステップ積分したもの（ネットの計算は 10 回）
    生徒: ノイズを入れると、行動を 1 回で出すネット（ネットの計算は 1 回）
    お手本のデータは使わない. 使うのは、教師に作らせた（ノイズ, 行動）の組だけ.
    """
    net = MLP([1, 64, 64, 1], rng)
    log = []
    for k in range(steps):
        eps = rng.standard_normal(batch)
        target = teacher(eps)
        log.append(net.train_step(eps[:, None], target[:, None], cosine_lr(k, steps, 3e-3)))
    return net, np.array(log)


# ---------------------------------------------------------------------------
# (5) 知識蒸留の温度
# ---------------------------------------------------------------------------
def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=float) / temp
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAPGothic", "IPAGothic",
                 "Hiragino Sans", "Yu Gothic", "Meiryo", "TakaoGothic", "VL Gothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[注意] 日本語フォントが見つかりません. 図の日本語が □ になるときは、"
              "sudo apt install fonts-noto-cjk を実行してください.")
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9AA3B0", "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": "#4A5260", "ytick.color": "#4A5260",
        "axes.grid": True, "grid.color": "#E3E6EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 2.0, "axes.unicode_minus": False,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 100,
    })
    return plt


def fig_paths(plt, rng, net):
    """上: 学習で使う直線（お手本とノイズの組）. 下: 学習した速度場に沿った軌道."""
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.9), sharex=True)
    tau = np.linspace(0, 1, 2)

    ax = axes[0]
    eps = rng.standard_normal(36)
    a = sample_actions(rng, 36)
    for e, x1 in zip(eps, a):
        ax.plot(tau, [e, x1], color=C_A if x1 < 0 else C_B, lw=1.1, alpha=0.75)
    ax.set_title("学習の目標：ノイズとお手本を結ぶ直線", loc="left")
    ax.set_ylabel("行動 a")
    ax.set_ylim(-2.6, 2.6)

    ax = axes[1]
    q = (np.arange(40) + 0.5) / 40
    eps = np.sqrt(2) * erfinv_approx(2 * q - 1)          # 標準正規分布の分位点（等確率に 40 本）
    end, path = integrate(lambda x, t: velocity(net, x, t), eps, 100, keep_path=True)
    ts = np.linspace(0, 1, path.shape[0])
    for j in range(path.shape[1]):
        ax.plot(ts, path[:, j], color=C_A if end[j] < 0 else C_B, lw=1.1, alpha=0.8)
    ax.set_title("推論：学習した速度場に沿った軌道", loc="left")
    ax.set_xlabel("フローの時間 τ（0 がノイズ、1 が行動）")
    ax.set_ylabel("行動 a")
    ax.set_ylim(-2.6, 2.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    handles = [plt.Line2D([], [], color=C_A, lw=2), plt.Line2D([], [], color=C_B, lw=2)]
    axes[0].legend(handles, ["左（a = −1）へ行くもの", "右（a = +1）へ行くもの"], loc="upper center",
                   bbox_to_anchor=(0.5, 1.42), ncol=1, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch10_paths.png")
    plt.close(fig)


def erfinv_approx(y):
    """誤差関数の逆関数（Winitzki の近似）. scipy を使わないための小さな代用品."""
    a = 0.147
    ln = np.log(1 - y**2)
    return np.sign(y) * np.sqrt(np.sqrt((2 / (np.pi * a) + ln / 2) ** 2 - ln / a) - (2 / (np.pi * a) + ln / 2))


def fig_samples(plt, rows):
    """取り出した行動の分布を、条件ごとに並べる."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(4.6, 1.22 * len(rows) + 0.7), sharex=True)
    bins = np.linspace(-2, 2, 81)
    for ax, (label, samples) in zip(axes, rows):
        ax.hist(samples, bins=bins, color=C_A, alpha=0.9)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        for m in MODES:
            ax.axvline(m, color=C_MUTED, lw=0.8, ls=":")
        ax.text(0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5)
        ax.text(1.0, 1.04, f"山に入った割合 {100 * in_mode(samples):.1f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="#4A5260")
    axes[-1].set_xlabel("行動 a（−1 が左へ避ける、+1 が右へ避ける）")
    axes[-1].set_xticks([-2, -1, 0, 1, 2])
    fig.tight_layout(h_pad=1.5)
    fig.savefig(FIG_DIR / "ch10_samples.png")
    plt.close(fig)


def fig_map(plt, net, student):
    """ノイズ ε から行動 a への写像. 教師（ステップ数ごと）と生徒."""
    eps = np.linspace(-2.5, 2.5, 501)
    vel = lambda x, t: velocity(net, x, t)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.plot(eps, integrate(vel, eps, 1), color=C_MUTED, lw=1.6, ls=":", label="教師 1 ステップ")
    ax.plot(eps, integrate(vel, eps, 2), color=C_MUTED, lw=1.6, ls="--", label="教師 2 ステップ")
    ax.plot(eps, integrate(vel, eps, 10), color=C_A, lw=5.0, alpha=0.45, solid_capstyle="round", label="教師 10 ステップ")
    ax.plot(eps, student(eps[:, None])[:, 0], color=C_B, lw=1.6, label="生徒（計算は 1 回）")
    ax.set_xlabel("はじめのノイズ ε")
    ax.set_ylabel("出てくる行動 a")
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper left", handlelength=2.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ch10_map.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_eval = 20000
    data = sample_actions(rng, n_eval)

    print("(1) 回帰: 二乗誤差で 1 つの答え c を学習する")
    c = float(np.mean(data))    # E[(a - c)^2] を最小にする c は、a の平均
    print(f"    最小にする c = お手本の平均 = {c:+.3f}")
    print(f"    お手本が山に入っている割合 {100 * in_mode(data):.1f}%、回帰の答えは {100 * in_mode(np.full(1, c)):.0f}%（正面に突っ込む）")

    print("\n(2) 学習: フローマッチング（6000 回の更新）")
    net, log = train_flow(rng)
    print(f"    損失: はじめの 100 回の平均 {log[:100].mean():.3f} → 終わりの 100 回の平均 {log[-100:].mean():.3f}")
    print("    損失は 0 にならない. 同じ点 a^τ を通る直線が何本もあり、目標 a - ε がばらつくため.")
    tau = rng.random(n_eval)
    x = (1 - tau) * rng.standard_normal(n_eval) + tau * sample_actions(rng, n_eval)
    err = velocity(net, x, tau) - exact_velocity(x, tau)
    print(f"    理論の式 v*(x, τ) = E[a - ε | a^τ = x] とのずれ（二乗平均の平方根）: {np.sqrt(np.mean(err**2)):.3f}"
          f"（v* の大きさは {np.sqrt(np.mean(exact_velocity(x, tau)**2)):.2f}）")
    print(f"    τ = 0 での確認: v(ε, 0) は −ε になるはず. ε = 1.0 で {velocity(net, np.array([1.0]), 0.0)[0]:+.3f}、"
          f"理論は {exact_velocity(np.array([1.0]), 0.0)[0]:+.3f}")

    print("\n(3) 推論: オイラー法のステップ数 N を変える（ネットの計算は N 回）")
    eps = rng.standard_normal(n_eval)
    vel = lambda x, t: velocity(net, x, t)
    samples = {}
    for n in (1, 2, 4, 10, 100):
        samples[n] = integrate(vel, eps, n)
        print(f"    N = {n:3d}: 山に入った割合 {100 * in_mode(samples[n]):5.1f}%、|a| の平均 {np.mean(np.abs(samples[n])):.3f}")
    print(f"    お手本        : 山に入った割合 {100 * in_mode(data):5.1f}%、|a| の平均 {np.mean(np.abs(data)):.3f}")

    print("\n(4) 蒸留: 10 ステップの教師 → 1 回で出す生徒（3000 回の更新）")
    teacher = lambda e: integrate(vel, e, 10)
    student, dlog = distill(rng, teacher)
    out = student(eps[:, None])[:, 0]
    print(f"    損失: はじめの 100 回の平均 {dlog[:100].mean():.4f} → 終わりの 100 回の平均 {dlog[-100:].mean():.6f}")
    print(f"    教師 10 ステップ（計算 10 回）: 山に入った割合 {100 * in_mode(samples[10]):5.1f}%")
    print(f"    教師  1 ステップ（計算  1 回）: 山に入った割合 {100 * in_mode(samples[1]):5.1f}%")
    print(f"    生徒            （計算  1 回）: 山に入った割合 {100 * in_mode(out):5.1f}%")
    print(f"    生徒と教師の出力のずれ（二乗平均の平方根）: {np.sqrt(np.mean((out - samples[10])**2)):.3f}")
    print(f"    左右の割合（生徒）: 左 {100 * np.mean(out < 0):.1f}%、右 {100 * np.mean(out > 0):.1f}%")

    print("\n(5) 知識蒸留の温度: 教師のロジット z = (4, 2, 0) を、温度 T の softmax に通す")
    for temp in (1.0, 2.0, 4.0):
        p = softmax([4.0, 2.0, 0.0], temp)
        print(f"    T = {temp:.0f}: p = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print("    T を上げると、正解でないクラスどうしの差（2 番目は 3 番目より近い）が見えやすくなる.")

    plt = setup_matplotlib()
    fig_paths(plt, np.random.default_rng(1), net)
    fig_samples(plt, [
        ("お手本", data),
        ("回帰（二乗誤差）", np.full(n_eval, c)),
        ("フロー 1 ステップ", samples[1]),
        ("フロー 2 ステップ", samples[2]),
        ("フロー 10 ステップ（教師）", samples[10]),
        ("蒸留した生徒（計算は 1 回）", out),
    ])
    fig_map(plt, net, student)
    print(f"\n図を保存しました: {FIG_DIR.relative_to(REPO_ROOT)}/ch10_paths.png, ch10_samples.png, ch10_map.png")


if __name__ == "__main__":
    main()
