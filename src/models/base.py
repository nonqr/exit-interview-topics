"""Общий интерфейс тематических моделей.

Все модели (LDA, NMF, эмбеддинги) принимают одно и то же:
    texts  — тексты после лёгкой очистки (для эмбеддингов);
    tokens — списки лемм после полной предобработки (для LDA/NMF и ключевых слов);
и возвращают TopicResult. Благодаря этому интерфейс, метрики и сравнение
моделей не зависят от того, какая модель внутри.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

OUTLIER = -1  # документ не отнесён ни к одной теме (бывает у HDBSCAN)


@dataclass
class TopicResult:
    labels: np.ndarray  # номер темы для каждого документа
    topic_words: list[list[tuple[str, float]]]  # для каждой темы: (слово, вес) по убыванию веса
    doc_topic: np.ndarray | None = None  # (n_docs, n_topics): уверенность в каждой теме

    @property
    def n_topics(self) -> int:
        return len(self.topic_words)

    def keywords(self, n: int = 10) -> list[list[str]]:
        return [[w for w, _ in words[:n]] for words in self.topic_words]

    def confidence(self) -> np.ndarray:
        """Насколько модель уверена в теме каждого документа (0…1)."""
        if self.doc_topic is None:
            return np.ones(len(self.labels))
        return self.doc_topic.max(axis=1)

    def topic_info(self, n_words: int = 8) -> pd.DataFrame:
        """Таблица тем: размер, доля, ключевые слова."""
        sizes = pd.Series(self.labels).value_counts()
        rows = [
            {"topic": t, "size": int(sizes.get(t, 0)), "share": sizes.get(t, 0) / len(self.labels),
             "keywords": ", ".join(self.keywords(n_words)[t])}
            for t in range(self.n_topics)
        ]
        info = pd.DataFrame(rows)
        if (self.labels == OUTLIER).any():
            n_out = int((self.labels == OUTLIER).sum())
            info = pd.concat([info, pd.DataFrame([{"topic": OUTLIER, "size": n_out,
                                                    "share": n_out / len(self.labels), "keywords": "— вне тем —"}])])
        return info.sort_values("size", ascending=False).reset_index(drop=True)

    def representative_docs(self, texts: Sequence[str], topic: int, n: int = 5) -> list[str]:
        """Самые типичные ответы темы — те, где модель уверена сильнее всего."""
        idx = np.where(self.labels == topic)[0]
        if self.doc_topic is not None and topic >= 0:
            idx = idx[np.argsort(-self.doc_topic[idx, topic])]
        return [texts[i] for i in idx[:n]]


class TopicModel(ABC):
    """Базовый класс.

    fit     — обучиться и вернуть темы для обучающих документов;
    predict — отнести новые документы к уже найденным темам (нужно для оценки
              устойчивости и для новых анкет без переобучения).
    """

    name: str = "model"

    @abstractmethod
    def fit(self, texts: Sequence[str], tokens: Sequence[Sequence[str]]) -> TopicResult: ...

    @abstractmethod
    def predict(self, texts: Sequence[str], tokens: Sequence[Sequence[str]]) -> np.ndarray: ...


def top_words(weights: np.ndarray, vocab: np.ndarray, n: int = 20) -> list[list[tuple[str, float]]]:
    """Матрица «тема × слово» → список топ-слов с весами для каждой темы."""
    result = []
    for row in weights:
        order = np.argsort(row)[::-1][:n]
        result.append([(str(vocab[i]), float(row[i])) for i in order if row[i] > 0])
    return result
