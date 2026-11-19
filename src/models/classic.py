"""Классические тематические модели на «мешке слов»: LDA и NMF.

LDA (Latent Dirichlet Allocation) — вероятностная модель: каждый документ — смесь
тем, каждая тема — распределение вероятностей над словами. Обучается на
частотах слов (CountVectorizer).

NMF (Non-negative Matrix Factorization) — линейная алгебра: матрицу TF-IDF
«документ × слово» раскладываем в произведение двух неотрицательных матриц
W (документ × тема) и H (тема × слово). На коротких текстах обычно стабильнее LDA.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.models.base import TopicModel, TopicResult, top_words


def _join(tokens: Sequence[Sequence[str]]) -> list[str]:
    return [" ".join(t) for t in tokens]


def _analyzer(doc: str) -> list[str]:
    # Леммы уже готовы: делим по пробелам. Стандартный токенизатор sklearn
    # разорвал бы «колл-центр» и «не_платить» и выкинул бы однобуквенные слова.
    return doc.split()


class LDATopicModel(TopicModel):
    name = "LDA"

    def __init__(self, n_topics: int = 10, doc_topic_prior: float | None = None,
                 topic_word_prior: float | None = None, min_df: int = 2, max_iter: int = 100, seed: int = 0):
        """doc_topic_prior (α): чем меньше, тем меньше тем в одном документе.
        Для коротких ответов, где обычно одна причина, полезно взять α < 1/n_topics.
        topic_word_prior (β): чем меньше, тем меньше слов «держит» тема.
        """
        self.n_topics, self.min_df, self.max_iter, self.seed = n_topics, min_df, max_iter, seed
        self.doc_topic_prior, self.topic_word_prior = doc_topic_prior, topic_word_prior

    def fit(self, texts, tokens) -> TopicResult:
        vec = CountVectorizer(analyzer=_analyzer, min_df=self.min_df)
        X = vec.fit_transform(_join(tokens))
        lda = LatentDirichletAllocation(
            n_components=self.n_topics, doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior, learning_method="batch",
            max_iter=self.max_iter, random_state=self.seed,
        ).fit(X)
        self.vec_, self.model_ = vec, lda
        doc_topic = lda.transform(X)
        topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        return TopicResult(doc_topic.argmax(1), top_words(topic_word, vec.get_feature_names_out()), doc_topic)

    def predict(self, texts, tokens) -> np.ndarray:
        return self.model_.transform(self.vec_.transform(_join(tokens))).argmax(1)


class NMFTopicModel(TopicModel):
    name = "NMF"

    def __init__(self, n_topics: int = 10, min_df: int = 2, max_df: float = 0.5, seed: int = 0):
        self.n_topics, self.min_df, self.max_df, self.seed = n_topics, min_df, max_df, seed

    def fit(self, texts, tokens) -> TopicResult:
        vec = TfidfVectorizer(analyzer=_analyzer, min_df=self.min_df, max_df=self.max_df, sublinear_tf=True)
        X = vec.fit_transform(_join(tokens))
        nmf = NMF(n_components=self.n_topics, init="nndsvda", max_iter=1000, random_state=self.seed)
        W = nmf.fit_transform(X)
        self.vec_, self.model_ = vec, nmf
        # Нормируем строки W в «долю каждой темы в документе». Пустые документы → равномерно.
        sums = W.sum(axis=1, keepdims=True)
        doc_topic = np.where(sums > 0, W / np.where(sums == 0, 1, sums), 1 / self.n_topics)
        return TopicResult(doc_topic.argmax(1), top_words(nmf.components_, vec.get_feature_names_out()), doc_topic)

    def predict(self, texts, tokens) -> np.ndarray:
        W = self.model_.transform(self.vec_.transform(_join(tokens)))
        return W.argmax(1)
