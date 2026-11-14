"""演習で共通に使うもの: 図の保存先、色、matplotlib の設定."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "vla-math" / "figures"

# 図の色. 3 系列まで、白地で見分けやすいことを確かめた色を使う.
C_INK = "#18222D"
C_MUTED = "#8A94A0"
C_A = "#3949AB"   # 青
C_B = "#C2570C"   # 橙
C_C = "#00897B"   # 緑がかった青
SERIES = [C_A, C_B, C_C]


def setup_matplotlib():
    """日本語のフォントを探し、図の見た目をそろえる. pyplot を返す."""
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return plt


def blue_map():
    """白から青へ、明るさだけが変わる色の並び（大きさを表す図に使う）."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#C5CAE9", C_A, "#1A237E"])


def saved(*names):
    """保存した図の名前を表示する."""
    rel = FIG_DIR.relative_to(REPO_ROOT)
    print("\n図を保存しました: " + ", ".join(f"{rel}/{n}" for n in names))
