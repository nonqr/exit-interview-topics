"""Графики дашборда (plotly) и облака слов.

Правила оформления:
- доли тем — один цвет: важна величина, а не различие тем;
- тема × сегмент — тепловая карта в оттенках одного цвета (светлее — меньше);
- динамика — отдельный маленький график на каждую тему (а не 10 линий разных цветов,
  которые невозможно различить);
- у каждого графика есть подсказка при наведении.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#2a78d6"
# Последовательная шкала синего (из проверенной палитры): светлый → тёмный.
BLUE_SCALE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13)


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=30, b=10), font=FONT,
                      hoverlabel=dict(font_size=13), showlegend=False)
    return fig


def topic_shares(table: pd.DataFrame) -> go.Figure:
    """Горизонтальные столбцы: доля каждой темы, самые частые сверху."""
    t = table.sort_values("Ответов")
    fig = go.Figure(go.Bar(
        x=t["Доля"] * 100, y=t["Тема"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
        text=[f"{p:.0f}%" for p in t["Доля"] * 100], textposition="outside", cliponaxis=False,
        customdata=t[["Ответов", "Ключевые слова"]].values,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% ответов (%{customdata[0]})<br>%{customdata[1]}<extra></extra>",
    ))
    fig.update_xaxes(title="Доля ответов, %", showgrid=True, gridcolor="rgba(128,128,128,0.2)", rangemode="tozero")
    fig.update_yaxes(title=None)
    return _layout(fig, height=max(300, 34 * len(t) + 60))


def segment_heatmap(share: pd.DataFrame, segment_name: str) -> go.Figure:
    """Тепловая карта: доля темы (столбец) среди ответов сегмента (строка)."""
    fig = go.Figure(go.Heatmap(
        z=share.values, x=list(share.columns), y=[str(i) for i in share.index],
        colorscale=[[i / (len(BLUE_SCALE) - 1), c] for i, c in enumerate(BLUE_SCALE)],
        xgap=2, ygap=2, text=[[f"{v:.0f}" for v in row] for row in share.values], texttemplate="%{text}",
        colorbar=dict(title="%", thickness=12),
        hovertemplate=f"{segment_name}: <b>%{{y}}</b><br>Тема: %{{x}}<br>%{{z:.1f}}% ответов сегмента<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-35, side="bottom")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=max(320, 36 * len(share) + 160))


def timeline_small_multiples(table: pd.DataFrame, max_topics: int = 12) -> go.Figure:
    """Динамика доли каждой темы: по маленькому графику на тему, общая шкала."""
    cols = list(table.columns[:max_topics])
    long = table[cols].reset_index().melt(id_vars="Период", var_name="Тема", value_name="Доля, %")
    wrap = 3
    fig = px.line(long, x="Период", y="Доля, %", facet_col="Тема", facet_col_wrap=wrap, markers=True,
                  facet_row_spacing=0.12, facet_col_spacing=0.05)
    fig.update_traces(line=dict(color=BLUE, width=2), marker=dict(size=7),
                      hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1], font=dict(size=12)))
    fig.update_yaxes(title=None, rangemode="tozero", gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(title=None, tickformat="%m.%y")
    rows = math.ceil(len(cols) / wrap)
    return _layout(fig, height=190 * rows + 40)


def k_scores_chart(scores: dict[int, float], chosen: int) -> go.Figure:
    """Силуэт для разного числа тем — почему выбрано именно это число."""
    ks, vals = list(scores), list(scores.values())
    fig = go.Figure(go.Scatter(x=ks, y=vals, mode="lines+markers", line=dict(color=BLUE, width=2),
                               marker=dict(size=8), hovertemplate="%{x} тем: силуэт %{y:.3f}<extra></extra>"))
    fig.add_vline(x=chosen, line_dash="dot", line_color="gray")
    fig.update_xaxes(title="Число тем", dtick=1)
    fig.update_yaxes(title="Силуэт (выше — лучше)")
    return _layout(fig, height=260)


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    # Шрифт с кириллицей: DejaVu Sans есть в составе matplotlib.
    try:
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return str(path) if path.exists() else None
    except ImportError:
        return None


def wordcloud_image(words: list[tuple[str, float]], width: int = 520, height: int = 260):
    """Облако слов темы: размер слова — его вес в теме (c-TF-IDF)."""
    from wordcloud import WordCloud

    freqs = {w: max(v, 1e-6) for w, v in words if v > 0}
    if not freqs:
        return None

    def color(*_, **kwargs):  # оттенки синего, чтобы облако не выглядело пёстрым
        return BLUE_SCALE[3 + (kwargs.get("random_state").randint(0, 4) if kwargs.get("random_state") else 0)]

    wc = WordCloud(width=width, height=height, background_color=None, mode="RGBA", font_path=_font_path(),
                   prefer_horizontal=0.95, max_words=30, color_func=color, random_state=0)
    return wc.generate_from_frequencies(freqs).to_image()
