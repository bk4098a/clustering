import matplotlib
matplotlib.use("Agg")   # 저장 전용
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, glob
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib import cycler
import matplotlib as mpl
from pathlib import Path as FsPath

# ---------- 출력 폴더 ----------
outdir = Path(r"C:/Users/KimByeolha/Downloads/cluster")
outdir.mkdir(parents=True, exist_ok=True)

# ---------- 한글 폰트 ----------
def use_korean_font(font_dir=r"C:/Users/KimByeolha/Downloads/프박뽑았체_TTF",
                    fallbacks=("Malgun Gothic", "Noto Sans CJK KR", "AppleGothic")):
    added = []
    if font_dir and os.path.isdir(font_dir):
        for p in glob.glob(os.path.join(font_dir, "*.ttf")):
            try:
                fm.fontManager.addfont(p); added.append(p)
            except Exception: pass
    chosen_name = None
    preferred_order = ["Medium","Bold","Light"]
    if added:
        for pref in preferred_order:
            cand = [p for p in added if pref.lower() in os.path.basename(p).lower()]
            if cand:
                try:
                    chosen_name = FontProperties(fname=cand[0]).get_name(); break
                except Exception: pass
        if not chosen_name:
            try: chosen_name = FontProperties(fname=added[0]).get_name()
            except Exception: chosen_name = None

    fams = []
    if chosen_name: fams.append(chosen_name)
    fams.extend(list(fallbacks))
    mpl.rcParams["font.family"] = fams[0] if fams else mpl.rcParams.get("font.family","sans-serif")
    mpl.rcParams["font.sans-serif"] = fams
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42; mpl.rcParams["ps.fonttype"] = 42; mpl.rcParams["svg.fonttype"] = "none"
    return chosen_name

use_korean_font()

try:
    fams = list(mpl.rcParams.get("font.sans-serif", []))
    for f in ["DejaVu Sans", "Arial Unicode MS", "Malgun Gothic", "Noto Sans CJK KR", "AppleGothic"]:
        if f not in fams:
            fams.append(f)
    mpl.rcParams["font.sans-serif"] = fams
except Exception:
    pass

# ---------- THEME: Navy × Sky ----------
def set_tridge_navy_sky(dark_slide=False):
    """
    dark_slide=True면 어두운 슬라이드(다크 배경)용으로 잉크/그리드 컬러를 반전.
    """
    brand = {
        # core blues
        "navy1": "#0B1A3C",   # 강한 포커스(타이틀·메인 라인)
        "navy2": "#123A6B",   # 축/엣지·서브 라인
        "blue" : "#2F5FB3",   # 기본 막대/라인
        "sky1" : "#60A5FF",   # 하이라이트
        "sky2" : "#9CC9FF",   # 보조(연한 하늘)
        # grayscale
        "ink"  : ("white" if dark_slide else "#111318"),
        "muted": ("#E6ECF7" if dark_slide else "#3C465A"),
        "grid" : ("#FFFFFF2A" if dark_slide else "#0B1A3C20"),
        # pos/neg
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

brand = set_tridge_navy_sky(dark_slide=False)

def polish_axes(ax):
    for side in ["top","right"]:
        ax.spines[side].set_visible(False)
    ax.margins(x=0.03)
    return ax

def bar_colors(n):
    base = [brand["sky2"], brand["sky1"], brand["blue"], brand["navy2"], brand["navy1"]]
    if n <= len(base): return base[:n]
    from itertools import cycle, islice
    return list(islice(cycle(base), n))

# ---------- 저장 ----------
import re, os
from pathlib import Path

# optional: keep a short fallback dir ready
fallback_dir = Path(r"C:/plots_tmp")
fallback_dir.mkdir(parents=True, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    # Replace Windows-forbidden chars; strip trailing dots/spaces
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", name)
    name = name.strip().rstrip(" .")
    return name or "plot.png"

def save_plot(fig, filename: str, w=11, h=6.2, dpi=360):
    # outdir / fallback_dir 이 문자열일 수도 있으니 안전하게 Path로
    _outdir = FsPath(outdir)
    _fallback_dir = FsPath(fallback_dir)

    # 파일명 정규화
    safe = _sanitize_filename(filename)
    target = _outdir / safe

    # 경로 길이 너무 길면 fallback
    full = str(target)
    if len(full) >= 240:
        target = _fallback_dir / safe
        full = str(target)

    # 디렉토리 보장 (★ pathlib 전용 별칭 사용)
    target.parent.mkdir(parents=True, exist_ok=True)

    # 사이즈/저장
    fig.set_size_inches(w, h)
    try:
        fig.savefig(full, dpi=dpi, bbox_inches="tight", transparent=True)
    except OSError as e:
        # 최종 백업: 파일명 짧게 만들어 fallback에 저장
        short = re.sub(r"[^A-Za-z0-9_.-]+", "", FsPath(safe).stem)[:24] + ".png"
        final = str(_fallback_dir / short)
        fig.savefig(final, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"[WARN] Saved to fallback: {final} (reason: {e!r})")
    finally:
        plt.close(fig)
        plt.close('all')



# ---------- 헬퍼 ----------
def annotate_bar(ax, fmt="{:,.0f}", dy=0.02, fontsize=10):
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x()+p.get_width()/2, h*(1+dy), fmt.format(h), ha="center", va="bottom", fontsize=fontsize)

# =========================
# 📌 Per-Category Visuals (matplotlib only, your theme reused)
# =========================
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 설정 ----
CLEANED_DIR = Path(r"C:\Users\KimByeolha\Downloads\cleaned")
USE_NORM    = True   # True: category_ingredient_matrix_norm.csv, False: category_ingredient_matrix.csv
INFILE      = CLEANED_DIR / ("category_ingredient_matrix_norm.csv" if USE_NORM else "category_ingredient_matrix.csv")

OUT_BASE    = outdir / "per_category"        # 최종 저장 폴더: plots_ITA_extra/per_category
OUT_BASE.mkdir(parents=True, exist_ok=True)

TOPN_DEFAULT = 20      # 카테고리별 상위 N개만 시각화(가독성)
MIN_SCORE    = 0.0     # 0은 제외됨(=같은 효과), 필요시 임계값을 소폭 올리면 더 깔끔

# ---------- 공통: 카테고리→시리즈 준비 ----------
def get_category_series(mat: pd.DataFrame, category: str, min_score=0.0, topn=TOPN_DEFAULT):
    """
    카테고리 한 줄을 (원재료, 점수) Series로 추출
    - 0 또는 min_score 이하 제거
    - 내림차순 정렬, 상위 topn만
    """
    if category not in mat.index:
        return pd.Series(dtype=float)
    s = mat.loc[category].copy()
    s = s[s > min_score].sort_values(ascending=False)
    if topn and len(s) > topn:
        s = s.iloc[:topn]
    return s

# ---------- 1) Bar (Top-N) ----------
def plot_category_bar(category: str, s: pd.Series):
    fig, ax = plt.subplots()
    cols = bar_colors(len(s))
    ax.barh(s.index[::-1], s.values[::-1], color=cols)
    ax.set_title(f"[{category}] 대표 원재료 Top-{len(s)}")
    ax.set_xlabel("점수")
    ax.set_ylabel("원재료")
    # 값 라벨
    for i, v in enumerate(s.values[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center")
    polish_axes(ax)
    save_plot(fig, str(OUT_BASE / f"{category}__bar_top{len(s)}.png"), w=10.5, h=max(5.2, 0.35*len(s)+2.2))

# ---------- 2) Lollipop ----------
def plot_category_lollipop(category: str, s: pd.Series):
    fig, ax = plt.subplots()
    y = np.arange(len(s))[::-1]
    ax.hlines(y=y, xmin=0, xmax=s.values[::-1], color=brand["grid"])
    ax.plot(s.values[::-1], y, "o", markersize=6, color=brand["blue"])
    ax.set_yticks(y, labels=s.index[::-1])
    ax.set_xlabel("점수")
    ax.set_title(f"[{category}] 원재료 Lollipop (Top-{len(s)})")
    # 값 라벨
    for xv, yv in zip(s.values[::-1], y):
        ax.text(xv, yv, f" {xv:.3f}", va="center")
    polish_axes(ax)
    save_plot(fig, str(OUT_BASE / f"{category}__lollipop_top{len(s)}.png"), w=10.5, h=max(5.0, 0.34*len(s)+2.0))

# ---------- 3) Polar Radial Bar (Ingredient Wheel) ----------
def plot_category_polar(category: str, s: pd.Series):
    """
    원형으로 펼친 레이디얼 바 – 프레젠테이션에 시각적 임팩트 좋음.
    """
    N = len(s)
    if N == 0:
        return
    # 각도/반지름
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii  = s.values
    labels = s.index.tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_direction(-1)     # 시계방향
    ax.set_theta_offset(np.pi/2)   # 12시 방향 시작
    cols = bar_colors(N)
    bars = ax.bar(angles, radii, width=2*np.pi/N*0.88, align="edge", color=cols, edgecolor=brand["navy2"])
    # 라벨(밖으로 살짝)
    for ang, r, lab in zip(angles, radii, labels):
        ax.text(ang + (np.pi/N)*0.44, r*1.05, lab, ha="center", va="bottom", rotation=np.degrees(-ang),
                rotation_mode="anchor", fontsize=9)
    ax.set_title(f"[{category}] Ingredient Wheel (Top-{N})", va="bottom")
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.grid(False)
    save_plot(fig, str(OUT_BASE / f"{category}__polar_wheel_top{N}.png"), w=8.6, h=8.6)

# ---------- 4) Micro Heat-Strip (1×N Heatmap) ----------
def plot_category_heatstrip(category: str, s: pd.Series):
    """
    1행짜리 마이크로 히트맵: 레이아웃에 끼워넣기 좋음.
    """
    if len(s) == 0:
        return
    arr = s.values.reshape(1, -1)
    fig, ax = plt.subplots()
    im = ax.imshow(arr, aspect="auto")
    ax.set_yticks([0], labels=[category])
    ax.set_xticks(np.arange(len(s)), labels=s.index, rotation=90)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("점수")
    polish_axes(ax)
    w = min(16, 1 + 0.30*len(s))
    save_plot(fig, str(OUT_BASE / f"{category}__heatstrip_top{len(s)}.png"), w=w, h=3.6)

# ---------- 배치 실행 ----------
def run_per_category(min_score=MIN_SCORE, topn=TOPN_DEFAULT, categories=None):
    # 매트릭스 로드
    mat = pd.read_csv(INFILE, index_col=0)
    # 대상 카테고리 선택
    cats = list(mat.index) if categories is None else [c for c in categories if c in mat.index]
    # 생성
    for cat in cats:
        s = get_category_series(mat, cat, min_score=min_score, topn=topn)
        if s.empty:
            continue
        # 0 제거는 get_category_series에서 이미 처리
        plot_category_bar(cat, s)
        plot_category_lollipop(cat, s)
        plot_category_polar(cat, s)
        plot_category_heatstrip(cat, s)
    print(f"[done] per-category plots saved in: {OUT_BASE}")

# ========== 실행 예시 ==========
if __name__ == "__main__":
    # 전체 카테고리에 대해 Top-20만, 0점 제외
    run_per_category(min_score=0.0, topn=20)
    # 특정 카테고리만 원하면:
    # run_per_category(min_score=0.0, topn=15, categories=["비스킷", "시리얼", "요거트"])
