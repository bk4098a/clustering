# -*- coding: utf-8 -*-
"""
Per-Category Ingredient Visuals (matplotlib-only, Navy × Sky theme)
- Input: category_ingredient_matrix(_norm).csv (index=카테고리, columns=원재료)
- Output: ./cluster_outputs/per_category/*.png (bar / lollipop / heat-strip)
"""

import os, re, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless 저장 전용
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib import cycler
import matplotlib as mpl

# =========================
# Theme / Font
# =========================
def use_korean_font(font_dir=None,
                    fallbacks=("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "DejaVu Sans")):
    added = []
    if font_dir and Path(font_dir).is_dir():
        for p in glob.glob(os.path.join(font_dir, "*.ttf")):
            try:
                fm.fontManager.addfont(p); added.append(p)
            except Exception:
                pass
    chosen_name = None
    if added:
        # Medium > Bold > Light 우선
        for pref in ("Medium","Bold","Light"):
            cand = [p for p in added if pref.lower() in os.path.basename(p).lower()]
            if cand:
                try:
                    chosen_name = FontProperties(fname=cand[0]).get_name()
                    break
                except Exception:
                    pass
        if not chosen_name:
            try:
                chosen_name = FontProperties(fname=added[0]).get_name()
            except Exception:
                chosen_name = None

    fams = []
    if chosen_name: fams.append(chosen_name)
    fams.extend(list(fallbacks))
    # rc 설정
    mpl.rcParams["font.family"] = fams[0] if fams else mpl.rcParams.get("font.family","sans-serif")
    mpl.rcParams["font.sans-serif"] = fams
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42; mpl.rcParams["ps.fonttype"] = 42; mpl.rcParams["svg.fonttype"] = "none"
    return chosen_name

def set_tridge_navy_sky(dark_slide=False):
    brand = {
        "navy1": "#0B1A3C",
        "navy2": "#123A6B",
        "blue" : "#2F5FB3",
        "sky1" : "#60A5FF",
        "sky2" : "#9CC9FF",
        "ink"  : ("white" if dark_slide else "#111318"),
        "muted": ("#E6ECF7" if dark_slide else "#3C465A"),
        "grid" : ("#FFFFFF2A" if dark_slide else "#0B1A3C20"),
        "pos"  : "#19C37D",
        "neg"  : "#FF5C51"
    }
    mpl.rcParams["axes.prop_cycle"] = cycler(color=[brand["blue"], brand["navy2"], brand["sky1"], brand["navy1"], brand["sky2"]])
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor":   "none",
        "text.color":       brand["ink"],
        "axes.labelcolor":  brand["ink"],
        "xtick.color":      brand["ink"],
        "ytick.color":      brand["ink"],
        "axes.edgecolor":   brand["muted"],
        "axes.linewidth":   1.1,
        "axes.titleweight": "bold",
        "font.size":        12.5,
        "axes.grid":        True,
        "grid.alpha":       1.0,
        "grid.color":       brand["grid"],
        "grid.linestyle":   (0, (3, 5)),
        "axes.grid.axis":   "y",
        "legend.frameon":   False,
        "legend.fontsize":  10.5,
        "pdf.fonttype":     42, "ps.fonttype": 42, "svg.fonttype": "none"
    })
    return brand

def polish_axes(ax):
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.03)
    return ax

def bar_colors(n, brand):
    base = [brand["sky2"], brand["sky1"], brand["blue"], brand["navy2"], brand["navy1"]]
    if n <= len(base): return base[:n]
    from itertools import cycle, islice
    return list(islice(cycle(base), n))

# =========================
# Save helper
# =========================
def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", name)
    name = name.strip().rstrip(" .")
    return name or "plot.png"

def save_plot(fig, path: Path, w=11, h=6.2, dpi=360):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(w, h)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig); plt.close('all')

# =========================
# Data helpers
# =========================
def get_category_series(mat: pd.DataFrame, category: str, min_score=0.0, topn=20):
    if category not in mat.index:
        return pd.Series(dtype=float)
    s = mat.loc[category].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = s[s > float(min_score)].sort_values(ascending=False)
    if topn and len(s) > topn:
        s = s.iloc[:topn]
    return s

# =========================
# Plots (bar / lollipop / heat-strip)
# =========================
def plot_category_bar(category: str, s: pd.Series, outdir: Path, brand):
    fig, ax = plt.subplots()
    cols = bar_colors(len(s), brand)
    ax.barh(s.index[::-1], s.values[::-1], color=cols)
    ax.set_title(f"[{category}] 대표 원재료 Top-{len(s)}")
    ax.set_xlabel("점수"); ax.set_ylabel("원재료")
    for i, v in enumerate(s.values[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center")
    polish_axes(ax)
    save_plot(fig, outdir / f"{_sanitize_filename(category)}__bar_top{len(s)}.png",
              w=10.5, h=max(5.2, 0.35*len(s)+2.2))

def plot_category_lollipop(category: str, s: pd.Series, outdir: Path, brand):
    fig, ax = plt.subplots()
    y = np.arange(len(s))[::-1]
    ax.hlines(y=y, xmin=0, xmax=s.values[::-1], color=brand["grid"])
    ax.plot(s.values[::-1], y, "o", markersize=6, color=brand["blue"])
    ax.set_yticks(y, labels=s.index[::-1])
    ax.set_xlabel("점수"); ax.set_title(f"[{category}] 원재료 Lollipop (Top-{len(s)})")
    for xv, yv in zip(s.values[::-1], y):
        ax.text(xv, yv, f" {xv:.3f}", va="center")
    polish_axes(ax)
    save_plot(fig, outdir / f"{_sanitize_filename(category)}__lollipop_top{len(s)}.png",
              w=10.5, h=max(5.0, 0.34*len(s)+2.0))

def plot_category_heatstrip(category: str, s: pd.Series, outdir: Path):
    if len(s) == 0:
        return
    arr = s.values.reshape(1, -1)
    fig, ax = plt.subplots()
    im = ax.imshow(arr, aspect="auto")
    ax.set_yticks([0], labels=[category])
    ax.set_xticks(np.arange(len(s)), labels=s.index, rotation=90)
    cb = fig.colorbar(im, ax=ax); cb.set_label("점수")
    polish_axes(ax)
    w = min(16, 1 + 0.30*len(s))
    save_plot(fig, outdir / f"{_sanitize_filename(category)}__heatstrip_top{len(s)}.png",
              w=w, h=3.6)

# =========================
# Runner (from DataFrame)
# =========================
def run_per_category_from_df(mat: pd.DataFrame,
                             outdir: Path,
                             brand: dict,
                             min_score=0.0,
                             topn=20,
                             categories=None):
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = list(mat.index) if categories is None else [c for c in categories if c in mat.index]
    per_dir = outdir / "per_category"
    per_dir.mkdir(parents=True, exist_ok=True)

    for cat in cats:
        s = get_category_series(mat, cat, min_score=min_score, topn=topn)
        if s.empty:
            continue
        plot_category_bar(cat, s, per_dir, brand)
        plot_category_lollipop(cat, s, per_dir, brand)
        plot_category_heatstrip(cat, s, per_dir)
    print(f"[done] saved: {per_dir}")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Per-Category visuals (bar/lollipop/heat-strip)")
    ap.add_argument("--data", required=True,
                    help="Path to category_ingredient_matrix(_norm).csv (or .csv.gz)")
    ap.add_argument("--outdir", default="cluster_outputs",
                    help="Output root directory (default: ./cluster_outputs)")
    ap.add_argument("--font-dir", default=None,
                    help="Directory containing custom Korean TTFs (optional)")
    ap.add_argument("--dark", action="store_true", help="Dark slide mode")
    ap.add_argument("--min-score", type=float, default=0.0, help="Exclude <= this score (default 0.0)")
    ap.add_argument("--topn", type=int, default=20, help="Top-N ingredients per category (default 20)")
    ap.add_argument("--cats", default=None,
                    help="Comma-separated category names to limit (default: all)")
    args = ap.parse_args()

    # Fonts/Theme
    use_korean_font(font_dir=args.font_dir)
    brand = set_tridge_navy_sky(dark_slide=args.dark)

    # Load matrix
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if str(data_path).endswith(".gz"):
        mat = pd.read_csv(data_path, compression="gzip", index_col=0)
    else:
        mat = pd.read_csv(data_path, index_col=0)

    # Categories filter
    categories = None
    if args.cats:
        categories = [c.strip() for c in args.cats.split(",") if c.strip()]

    # Run
    outdir = Path(args.outdir)
    run_per_category_from_df(mat, outdir=outdir, brand=brand,
                             min_score=args.min_score, topn=args.topn, categories=categories)

if __name__ == "__main__":
    main()
