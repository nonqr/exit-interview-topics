"""Графики для документации и презентации."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Палитра (проверена на различимость при дальтонизме): синий — эмбеддинги, оранжевый — мешок слов.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
EMBEDDINGS = "#2a78d6"
CLASSIC = "#eb6834"

LABELS = {  # подписываем только ключевые точки, чтобы не загромождать график
    "LDA": "LDA",
    "NMF": "NMF",
    "GSDMM": "GSDMM",
    "multilingual-e5-large-instruct + hdbscan + UMAP5": "e5-instruct + HDBSCAN",
    "multilingual-e5-large-instruct + kmeans": "e5-instruct + KMeans",
    "multilingual-e5-large + kmeans": "e5-large + KMeans",
}


def plot_coherence_vs_ari(results: pd.DataFrame, path: str | Path | None = None):
    """Связность тем (без меток) против совпадения с эталоном (ARI).

    Главная мысль: модель с самой высокой связностью — LDA — почти не совпадает
    с реальными причинами. Выбирать модель только по связности нельзя.
    """
    df = results.copy()
    df["семейство"] = df["модель"].map(lambda m: "Эмбеддинги" if "+" in m else "Мешок слов")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for family, color in (("Мешок слов", CLASSIC), ("Эмбеддинги", EMBEDDINGS)):
        part = df[df["семейство"] == family]
        ax.scatter(part["c_v"], part["ARI"], s=64, color=color, edgecolor=SURFACE, linewidth=2,
                   label=family, zorder=3)
    for _, row in df.iterrows():
        if row["модель"] in LABELS:
            ax.annotate(LABELS[row["модель"]], (row["c_v"], row["ARI"]), xytext=(7, 4),
                        textcoords="offset points", fontsize=9, color=TEXT_2)

    ax.set_xlabel("Связность тем c_v — считается без эталона", color=TEXT_2, fontsize=10)
    ax.set_ylabel("ARI — совпадение с настоящими причинами", color=TEXT_2, fontsize=10)
    ax.set_title("Связность тем не показывает, насколько темы верны", loc="left", color=TEXT,
                 fontsize=13, pad=12)
    ax.set_ylim(-0.02, max(0.6, df["ARI"].max() + 0.05))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    legend = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor=SURFACE)
    return fig
