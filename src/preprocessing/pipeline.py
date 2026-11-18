"""Предобработка всей таблицы: очистка, фильтрация пустых ответов и дублей.

Результат — таблица с дополнительными колонками:
    clean_text — текст после лёгкой очистки (для эмбеддингов e5);
    tokens     — список лемм (для LDA / NMF);
    lemmas     — те же леммы строкой через пробел;
и отчёт о том, сколько строк и почему было отброшено.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from src.preprocessing.text import TextPreprocessor, normalize_text

# Колонки-идентификаторы анкеты: при поиске дублей их не сравниваем,
# потому что у повторной отправки той же анкеты номер новый.
ID_COLUMN_RE = re.compile(r"^(id|uid|uuid|№|n|номер.*|.*_id|id_.*|row_?id|index)$", re.I)


@dataclass
class PreprocessingReport:
    n_input: int
    n_empty: int  # пустые и бессмысленные ответы
    n_duplicates: int  # точные дубли строк (повторная отправка анкеты)
    n_output: int
    vocab_size: int  # число уникальных лемм
    avg_tokens: float  # среднее число лемм в ответе
    top_lemmas: list[tuple[str, int]] = field(default_factory=list)

    def as_text(self) -> str:
        top = ", ".join(f"{w} ({c})" for w, c in self.top_lemmas[:15])
        return (
            f"Строк на входе:           {self.n_input}\n"
            f"Пустые/бессмысленные:     {self.n_empty}\n"
            f"Точные дубли:             {self.n_duplicates}\n"
            f"Строк на выходе:          {self.n_output}\n"
            f"Размер словаря (лемм):    {self.vocab_size}\n"
            f"Среднее лемм в ответе:    {self.avg_tokens:.1f}\n"
            f"Самые частые леммы:       {top}"
        )


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    preprocessor: TextPreprocessor | None = None,
    drop_duplicates: bool = True,
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Очищает таблицу. Исходные колонки сохраняются без изменений."""
    tp = preprocessor or TextPreprocessor()
    out = df.copy()
    n_input = len(out)

    # Дубли ищем по всей строке: одинаковый текст у разных людей — это нормально,
    # а полностью одинаковая строка — повторная отправка анкеты.
    n_duplicates = 0
    if drop_duplicates:
        compare = [c for c in out.columns if not ID_COLUMN_RE.match(str(c).strip())] or list(out.columns)
        mask = out.duplicated(subset=compare, keep="first")
        n_duplicates = int(mask.sum())
        out = out[~mask]

    out["clean_text"] = out[text_col].map(normalize_text)
    out["is_meaningful"] = out[text_col].map(tp.is_meaningful)
    n_empty = int((~out["is_meaningful"]).sum())
    if drop_empty:
        out = out[out["is_meaningful"]]

    out["tokens"] = out[text_col].map(tp.tokens)
    out["lemmas"] = out["tokens"].map(" ".join)
    out = out.reset_index(drop=True)

    counts = Counter(t for toks in out["tokens"] for t in toks)
    report = PreprocessingReport(
        n_input=n_input,
        n_empty=n_empty,
        n_duplicates=n_duplicates,
        n_output=len(out),
        vocab_size=len(counts),
        avg_tokens=float(out["tokens"].map(len).mean()) if len(out) else 0.0,
        top_lemmas=counts.most_common(30),
    )
    return out, report
