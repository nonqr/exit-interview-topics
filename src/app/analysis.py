"""Логика веб-сервиса без Streamlit: анализ файла, темы, срезы, выгрузка.

Всё, что считается, живёт здесь — так это можно тестировать и переиспользовать.
streamlit_app.py только показывает результат и собирает ввод пользователя.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.models import EmbeddingTopicModel, Encoder, NMFTopicModel, TopicModel, TopicResult
from src.models.embedding import DEFAULT_MODEL, LIGHT_MODEL
from src.preprocessing import PreprocessingReport, TextPreprocessor, preprocess_dataframe

ROOT = Path(__file__).resolve().parents[2]
DEMO_FILE = ROOT / "data" / "raw" / "exit_interviews.xlsx"

# Режимы анализа, которые видит пользователь.
MODES = {
    "Точный (Qwen3-Embedding-4B)": DEFAULT_MODEL,
    "Быстрый (e5-large-instruct)": LIGHT_MODEL,
    "Без нейросети (NMF)": None,
}
DEFAULT_MODE = next(iter(MODES))

TOPIC_COL, NAME_COL, CONF_COL = "topic", "Тема", "Уверенность"
LOW_CONFIDENCE = 0.5  # ниже — ответ стоит перепроверить вручную


@dataclass
class Analysis:
    """Результат анализа одного файла."""

    data: pd.DataFrame  # исходные строки (без пустых и дублей) + тема, название, уверенность
    text_col: str
    result: TopicResult
    report: PreprocessingReport
    names: dict[int, str]  # номер темы → название (можно менять в интерфейсе)
    mode: str
    k_scores: dict[int, float] = field(default_factory=dict)  # силуэт для каждого k, если подбирали

    def topic_table(self) -> pd.DataFrame:
        """Темы, отсортированные по размеру: название, число ответов, доля, ключевые слова."""
        counts = self.data[NAME_COL].value_counts()
        kw = {self.names[t]: ", ".join(self.result.keywords(8)[t]) for t in range(self.result.n_topics)}
        table = pd.DataFrame({
            "Тема": counts.index,
            "Ответов": counts.values,
            "Доля": counts.values / counts.sum(),
        })
        # После объединения у одного названия несколько исходных тем — склеиваем их слова.
        table["Ключевые слова"] = table["Тема"].map(lambda n: _merged_keywords(self, n) or kw.get(n, ""))
        return table.reset_index(drop=True)

    def examples(self, name: str, n: int = 5) -> list[str]:
        """Самые типичные ответы темы (где модель уверена сильнее всего)."""
        part = self.data[self.data[NAME_COL] == name].sort_values(CONF_COL, ascending=False)
        return part[self.text_col].head(n).tolist()


def _merged_keywords(analysis: Analysis, name: str) -> str:
    topics = [t for t, n in analysis.names.items() if n == name]
    if len(topics) < 2:
        return ""
    words: list[str] = []
    for t in topics:
        for w in analysis.result.keywords(8)[t]:
            if w not in words:
                words.append(w)
    return ", ".join(words[:10])


# --- запуск анализа -------------------------------------------------------------


def make_model(mode: str, n_topics: int) -> TopicModel:
    model_name = MODES[mode]
    if model_name is None:
        return NMFTopicModel(n_topics=n_topics)
    return EmbeddingTopicModel(n_topics=n_topics, encoder=Encoder(model_name), clustering="kmeans")


def suggest_n_topics(embeddings: np.ndarray, k_range: range = range(5, 16)) -> tuple[int, dict[int, float]]:
    """Подбор числа тем по силуэтному коэффициенту.

    Силуэт показывает, насколько ответы ближе к своей теме, чем к соседней.
    На нашем датасете он выбирает 13 тем (ARI 0.64) при настоящих 10 (ARI 0.68),
    то есть это хорошая подсказка, но не истина: пользователь может поправить.
    """
    scores = {}
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(embeddings)
        scores[k] = float(silhouette_score(embeddings, labels, metric="cosine"))
    return max(scores, key=scores.get), scores


def default_names(result: TopicResult) -> dict[int, str]:
    """Название темы по умолчанию — три главных ключевых слова."""
    names = {}
    for t, words in enumerate(result.keywords(3)):
        base = ", ".join(words).capitalize() if words else f"Тема {t + 1}"
        name, i = base, 2
        while name in names.values():  # названия должны быть уникальными
            name, i = f"{base} ({i})", i + 1
        names[t] = name
    return names


def run_analysis(
    df: pd.DataFrame,
    text_col: str,
    mode: str = DEFAULT_MODE,
    n_topics: int | None = 10,
    extra_stopwords: tuple[str, ...] = (),
    progress: Callable[[str], None] = lambda msg: None,
    model: TopicModel | None = None,
) -> Analysis:
    """Полный цикл: очистка → эмбеддинги/леммы → темы. n_topics=None — подобрать автоматически."""
    progress("Очистка и лемматизация текста…")
    tp = TextPreprocessor(extra_stopwords=extra_stopwords)
    clean, report = preprocess_dataframe(df, text_col, tp)
    if len(clean) < 10:
        raise ValueError(f"После очистки осталось {len(clean)} ответов — слишком мало для анализа (нужно от 10).")
    texts, tokens = clean["clean_text"].tolist(), clean["tokens"].tolist()

    k_scores: dict[int, float] = {}
    if n_topics is None:
        progress("Подбор числа тем…")
        if MODES.get(mode) is None and model is None:
            n_topics = 10  # для NMF силуэт по эмбеддингам недоступен
        else:
            enc = model.encoder if isinstance(model, EmbeddingTopicModel) else Encoder(MODES[mode])
            n_topics, k_scores = suggest_n_topics(enc.encode(texts))
    n_topics = max(2, min(n_topics, len(clean) // 3))

    progress("Поиск тем… (первый запуск скачивает модель — это может занять несколько минут)")
    model = model or make_model(mode, n_topics)
    model.n_topics = n_topics
    result = model.fit(texts, tokens)

    names = default_names(result)
    data = clean.drop(columns=["tokens", "is_meaningful"]).copy()
    data[TOPIC_COL] = result.labels
    data[NAME_COL] = data[TOPIC_COL].map(names)
    data[CONF_COL] = result.confidence().round(3)
    return Analysis(data, text_col, result, report, names, mode, k_scores)


# --- правки пользователя ---------------------------------------------------------


def apply_names(analysis: Analysis, names: dict[int, str]) -> Analysis:
    """Переименование и объединение тем: темы с одинаковым названием считаются одной."""
    names = {t: (n.strip() or analysis.names[t]) for t, n in names.items()}
    analysis.names = names
    analysis.data[NAME_COL] = analysis.data[TOPIC_COL].map(names)
    return analysis


# --- срезы для дашборда ------------------------------------------------------------


def segment_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Колонки, по которым имеет смысл делать срез: от 2 до 30 разных значений."""
    cols = []
    for c in df.columns:
        if c in exclude or c in (TOPIC_COL, NAME_COL, CONF_COL, "clean_text", "lemmas"):
            continue
        n = df[c].nunique()
        if 2 <= n <= 30 and n < len(df) * 0.5:
            cols.append(c)
    return cols


def date_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Колонки, где почти все значения — даты."""
    cols = []
    for c in df.columns:
        if c in exclude or c in (TOPIC_COL, NAME_COL, CONF_COL):
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed", dayfirst=False)
        if parsed.notna().mean() > 0.9 and df[c].astype(str).str.contains(r"\d{4}|\d{2}[./-]\d{2}").mean() > 0.9:
            cols.append(c)
    return cols


def share_by_segment(analysis: Analysis, segment: str) -> pd.DataFrame:
    """Доля каждой темы внутри каждого сегмента (строки — сегменты, столбцы — темы), в процентах."""
    table = pd.crosstab(analysis.data[segment], analysis.data[NAME_COL], normalize="index") * 100
    order = analysis.data[NAME_COL].value_counts().index
    return table[[c for c in order if c in table.columns]]


def lift_by_segment(analysis: Analysis, segment: str) -> pd.DataFrame:
    """Где причина встречается заметно чаще обычного.

    Подъём = доля темы в сегменте / доля темы в среднем: 2.0 значит «в этом отделе
    причина встречается вдвое чаще». Но в маленьком сегменте большой подъём может
    быть случайностью, поэтому сортируем по стандартизированному остатку
    z = (наблюдаемое − ожидаемое) / √ожидаемое, как в критерии хи-квадрат.
    z ≥ 2 — отклонение, которое вряд ли объясняется случайностью.
    """
    data = analysis.data
    counts = pd.crosstab(data[segment], data[NAME_COL])
    seg_sizes = counts.sum(axis=1)
    overall = counts.sum(axis=0) / counts.values.sum()
    rows = []
    for seg in counts.index:
        for topic in counts.columns:
            observed = int(counts.loc[seg, topic])
            expected = seg_sizes[seg] * overall[topic]
            rows.append({"Сегмент": seg, "Тема": topic, "Ответов": observed, "Размер сегмента": int(seg_sizes[seg]),
                         "Доля в сегменте": observed / seg_sizes[seg], "Доля в среднем": overall[topic],
                         "Подъём": observed / expected if expected else 0.0,
                         "z": (observed - expected) / np.sqrt(expected) if expected else 0.0})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


def notable(insights: pd.DataFrame, z_min: float = 2.0, lift_min: float = 1.3, top: int = 5) -> pd.DataFrame:
    """Отклонения, которые стоит показать HR: статистически заметные и не мелкие."""
    return insights[(insights["z"] >= z_min) & (insights["Подъём"] >= lift_min)].head(top)


def timeline(analysis: Analysis, date_col: str, freq: str = "QS") -> pd.DataFrame:
    """Доля тем по периодам (по умолчанию — кварталы), в процентах."""
    dates = pd.to_datetime(analysis.data[date_col], errors="coerce", format="mixed")
    period = dates.dt.to_period(freq[0]).dt.start_time
    table = pd.crosstab(period, analysis.data[NAME_COL], normalize="index") * 100
    table.index.name = "Период"
    return table


# --- выгрузка ---------------------------------------------------------------------


def to_excel(analysis: Analysis) -> bytes:
    """Excel с тремя листами: ответы с темами, сводка по темам, отчёт об очистке."""
    buf = io.BytesIO()
    answers = analysis.data.drop(columns=[TOPIC_COL, "clean_text", "lemmas"], errors="ignore")
    r = analysis.report
    report = pd.DataFrame({
        "Показатель": ["Строк в файле", "Пустые и бессмысленные ответы", "Повторные анкеты",
                       "Проанализировано ответов", "Режим анализа"],
        "Значение": [r.n_input, r.n_empty, r.n_duplicates, r.n_output, analysis.mode],
    })
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        answers.to_excel(writer, sheet_name="Ответы", index=False)
        analysis.topic_table().to_excel(writer, sheet_name="Темы", index=False)
        report.to_excel(writer, sheet_name="Отчёт", index=False)
    return buf.getvalue()
