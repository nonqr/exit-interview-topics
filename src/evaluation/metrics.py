"""Метрики качества тематической модели.

Отвечают на вопрос «почему выходу модели можно верить». Четыре группы проверок:

1. Совпадение с эталоном (есть только на синтетике, где причина известна):
   ARI, NMI, чистота (purity), точность и полнота по каждой причине.
2. Внутреннее качество тем (работает и на реальных данных без меток):
   связность тем (coherence c_v и NPMI), разнообразие тем (topic diversity).
3. Устойчивость: модель, обученная заново на других подвыборках и с другим
   seed, должна находить те же темы. Неустойчивые темы — случайный шум.
4. Оценка людьми: тест «лишнее слово» (word intrusion) и согласованность
   разметчиков (каппа Коэна).

Все функции принимают простые типы: списки меток, списки слов тем, списки токенов
документов. Поэтому их можно применять к любой модели: LDA, NMF или BERTopic.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    normalized_mutual_info_score,
)

# ---------------------------------------------------------------------------
# 1. Совпадение с эталонными метками
# ---------------------------------------------------------------------------


def contingency(true: Sequence, pred: Sequence) -> pd.DataFrame:
    """Таблица сопряжённости: строки — эталонные причины, столбцы — темы модели."""
    return pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="pred"))


def majority_mapping(true: Sequence, pred: Sequence) -> dict:
    """Каждой теме модели сопоставляет причину, которая в ней встречается чаще всего.

    Тем у модели может быть больше или меньше, чем причин, поэтому используем
    соответствие «многие к одному». Оно нужно, чтобы назвать тему и посчитать
    точность и полноту.
    """
    table = contingency(true, pred)
    return {topic: table[topic].idxmax() for topic in table.columns}


def hungarian_mapping(true: Sequence, pred: Sequence) -> dict:
    """Взаимно однозначное сопоставление тем и причин (венгерский алгоритм).

    Строже, чем majority_mapping: две темы модели не могут «забрать» одну и ту же
    причину. Темы без пары не попадают в словарь.
    """
    table = contingency(true, pred)
    rows, cols = linear_sum_assignment(-table.values)  # максимизируем совпадения
    return {table.columns[c]: table.index[r] for r, c in zip(rows, cols)}


def purity(true: Sequence, pred: Sequence) -> float:
    """Доля документов, попавших в тему вместе с «большинством» своей причины."""
    table = contingency(true, pred)
    return float(table.max(axis=0).sum() / table.values.sum())


def per_label_report(true: Sequence, pred: Sequence, mapping: dict | None = None) -> pd.DataFrame:
    """Точность, полнота и F1 по каждой причине после сопоставления тем с причинами.

    precision — какая доля документов, отнесённых к причине, действительно про неё;
    recall    — какая доля документов про причину была найдена.
    """
    mapping = mapping if mapping is not None else majority_mapping(true, pred)
    predicted_label = pd.Series(pred).map(mapping)
    true = pd.Series(true).reset_index(drop=True)
    rows = []
    for label in sorted(true.unique()):
        tp = int(((predicted_label == label) & (true == label)).sum())
        n_pred = int((predicted_label == label).sum())
        n_true = int((true == label).sum())
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_true if n_true else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"label": label, "precision": p, "recall": r, "f1": f1, "support": n_true})
    return pd.DataFrame(rows).set_index("label")


def external_metrics(true: Sequence, pred: Sequence) -> dict[str, float]:
    """Сводные метрики совпадения с эталоном.

    ARI (adjusted Rand index): 1 — полное совпадение разбиений, 0 — как у случайного.
    NMI (normalized mutual information): 0…1, сколько информации о причине несёт тема.
    """
    report = per_label_report(true, pred)
    return {
        "ARI": adjusted_rand_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "purity": purity(true, pred),
        "macro_F1": float(report["f1"].mean()),
    }


def random_baseline(true: Sequence, n_topics: int, n_runs: int = 20, seed: int = 0) -> dict[str, float]:
    """Метрики случайного разбиения на n_topics тем — нижняя граница.

    Модель, которая не обгоняет этот бейзлайн с запасом, ничего не нашла.
    """
    rng = np.random.default_rng(seed)
    runs = [external_metrics(true, rng.integers(0, n_topics, len(true))) for _ in range(n_runs)]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


# ---------------------------------------------------------------------------
# 2. Внутреннее качество тем (без меток)
# ---------------------------------------------------------------------------


def topic_diversity(topics: Sequence[Sequence[str]], top_n: int = 10) -> float:
    """Доля уникальных слов среди топ-N слов всех тем.

    1.0 — темы не пересекаются по словам; около 0 — все темы одинаковые.
    """
    words = [w for topic in topics for w in list(topic)[:top_n]]
    return len(set(words)) / len(words) if words else 0.0


def coherence(
    topics: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    measure: str = "c_v",
    top_n: int = 10,
) -> tuple[float, list[float]]:
    """Связность тем: насколько часто топ-слова темы встречаются вместе в текстах.

    measure: "c_v" (0…1, чем выше, тем лучше; стандартная метрика в статьях)
             или "c_npmi" (-1…1; 0 — слова встречаются вместе не чаще случайного).
    Возвращает среднее значение и значение для каждой темы. Тему с низкой
    связностью стоит показать человеку: скорее всего, это мусорная тема.
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    dictionary = Dictionary(tokenized_docs)
    # gensim падает на словах, которых нет в словаре, — отфильтровываем.
    clean_topics = [[w for w in list(t)[:top_n] if w in dictionary.token2id] for t in topics]
    clean_topics = [t for t in clean_topics if len(t) >= 2]
    cm = CoherenceModel(
        topics=clean_topics, texts=list(tokenized_docs), dictionary=dictionary,
        coherence=measure, processes=1,
    )
    per_topic = cm.get_coherence_per_topic()
    return float(np.mean(per_topic)), [float(x) for x in per_topic]


# ---------------------------------------------------------------------------
# 3. Устойчивость
# ---------------------------------------------------------------------------


def stability(
    fit_predict: Callable[[Sequence[int], int], np.ndarray],
    n_docs: int,
    n_runs: int = 5,
    sample_frac: float = 0.8,
    seed: int = 0,
) -> dict[str, float]:
    """Устойчивость разбиения к случайности и к составу данных.

    fit_predict(indices, seed) должна обучить модель на документах с номерами
    indices и вернуть темы для ВСЕХ n_docs документов. Сравниваем разбиения
    разных запусков попарно через ARI: если результат сильно меняется от
    запуска к запуску, выводы по нему делать нельзя.
    """
    rng = np.random.default_rng(seed)
    labelings = []
    for run in range(n_runs):
        idx = np.sort(rng.choice(n_docs, size=int(n_docs * sample_frac), replace=False))
        labelings.append(np.asarray(fit_predict(idx, run)))
    pair_ari = [adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]
    return {"stability_ARI_mean": float(np.mean(pair_ari)), "stability_ARI_min": float(np.min(pair_ari))}


# ---------------------------------------------------------------------------
# 4. Оценка людьми
# ---------------------------------------------------------------------------


def word_intrusion_tasks(topics: Sequence[Sequence[str]], n_words: int = 5, seed: int = 0) -> pd.DataFrame:
    """Задания для теста «найди лишнее слово» (Chang et al., 2009).

    К топ-n_words словам темы подмешивается одно слово из топа другой темы.
    Если люди легко находят лишнее слово, тема осмысленная. Метрика —
    доля правильно найденных лишних слов (model precision).
    Колонку intruder перед раздачей разметчикам нужно скрыть.
    """
    rng = random.Random(seed)
    rows = []
    for i, topic in enumerate(topics):
        own = list(topic)[:n_words]
        all_own = set(topic)
        candidates = [w for j, other in enumerate(topics) if j != i for w in list(other)[:3] if w not in all_own]
        if not candidates:
            continue
        intruder = rng.choice(candidates)
        words = own + [intruder]
        rng.shuffle(words)
        rows.append({"topic": i, "words": ", ".join(words), "intruder": intruder})
    return pd.DataFrame(rows)


def intrusion_precision(tasks: pd.DataFrame, answers: Sequence[str]) -> float:
    """Доля заданий, где человек выбрал настоящее лишнее слово."""
    return float(np.mean([a.strip().lower() == t.lower() for a, t in zip(answers, tasks["intruder"])]))


def annotator_agreement(labels_a: Sequence, labels_b: Sequence) -> float:
    """Каппа Коэна между двумя разметчиками.

    Показывает, насколько сама задача однозначна: если люди согласны друг с
    другом на 0.6, требовать от модели 0.9 бессмысленно. Шкала:
    <0.4 слабое согласие, 0.4–0.6 умеренное, 0.6–0.8 хорошее, >0.8 отличное.
    """
    return float(cohen_kappa_score(labels_a, labels_b))
