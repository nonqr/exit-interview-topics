"""GSDMM — тематическая модель для коротких текстов (Yin & Wang, 2014).

Почему не LDA. LDA считает каждый документ смесью нескольких тем и оценивает
эту смесь по словам документа. В ответе из 5 слов данных для такой оценки
слишком мало, и модель становится неустойчивой.

GSDMM (Gibbs Sampling Dirichlet Multinomial Mixture) исходит из того, что у
короткого текста ровно одна тема. Обучение — сэмплирование по Гиббсу; авторы
описывают его как «процесс выбора столика в кинотеатре»: каждый документ
по очереди пересаживается за столик (в тему), где
  * уже сидит много документов (параметр α отвечает за эту «популярность»),
  * у документов похожие слова (параметр β отвечает за «похожесть»).
Число тем K задаётся как верхняя граница: лишние темы со временем пустеют,
так что модель сама находит число тем.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import gammaln

from src.models.base import TopicModel, TopicResult, top_words


class GSDMMTopicModel(TopicModel):
    name = "GSDMM"

    def __init__(self, n_topics: int = 20, alpha: float = 0.1, beta: float = 0.1,
                 n_iter: int = 40, min_df: int = 2, seed: int = 0):
        """n_topics — максимальное число тем; итоговых тем обычно меньше.
        Берите с запасом (в 1.5–2 раза больше ожидаемого): если максимум равен
        настоящему числу тем, две темы могут слиться в одну.
        """
        self.n_topics, self.alpha, self.beta = n_topics, alpha, beta
        self.n_iter, self.min_df, self.seed = n_iter, min_df, seed

    # --- вспомогательные функции ----------------------------------------------

    def _build_vocab(self, tokens: Sequence[Sequence[str]]) -> None:
        df: dict[str, int] = {}
        for doc in tokens:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        self.vocab_ = np.array(sorted(w for w, c in df.items() if c >= self.min_df))
        self.word_id_ = {w: i for i, w in enumerate(self.vocab_)}

    def _encode(self, tokens: Sequence[Sequence[str]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Документ → (номера слов, сколько раз каждое встречается)."""
        out = []
        for doc in tokens:
            ids = [self.word_id_[w] for w in doc if w in self.word_id_]
            uniq, counts = np.unique(np.array(ids, dtype=int), return_counts=True)
            out.append((uniq, counts))
        return out

    def _log_prob(self, doc: tuple[np.ndarray, np.ndarray], m_z, n_z, n_zw, n_docs: int) -> np.ndarray:
        """log p(тема | документ) для всех тем сразу, без нормировки (формула 4 из статьи)."""
        ids, counts = doc
        V, K = len(self.vocab_), len(m_z)
        log_p = np.log(m_z + self.alpha) - np.log(n_docs - 1 + K * self.alpha)
        if len(ids):
            # Π_w Π_{j=1..N_w} (n_zw + β + j − 1)  =  Γ(n_zw + β + N_w) / Γ(n_zw + β)
            nzw = n_zw[:, ids]
            log_p += (gammaln(nzw + self.beta + counts) - gammaln(nzw + self.beta)).sum(axis=1)
            # Π_{i=1..N_d} (n_z + Vβ + i − 1)  =  Γ(n_z + Vβ + N_d) / Γ(n_z + Vβ)
            n_d = counts.sum()
            log_p -= gammaln(n_z + V * self.beta + n_d) - gammaln(n_z + V * self.beta)
        return log_p

    @staticmethod
    def _softmax(log_p: np.ndarray) -> np.ndarray:
        p = np.exp(log_p - log_p.max())
        return p / p.sum()

    # --- обучение --------------------------------------------------------------

    def fit(self, texts, tokens) -> TopicResult:
        rng = np.random.default_rng(self.seed)
        self._build_vocab(tokens)
        docs = self._encode(tokens)
        D, K, V = len(docs), self.n_topics, len(self.vocab_)

        m_z = np.zeros(K)  # документов в теме
        n_z = np.zeros(K)  # слов в теме
        n_zw = np.zeros((K, V))  # сколько раз слово w встречается в теме z
        z = rng.integers(0, K, D)
        for d, (ids, counts) in enumerate(docs):
            m_z[z[d]] += 1
            n_z[z[d]] += counts.sum()
            n_zw[z[d], ids] += counts

        for _ in range(self.n_iter):
            moved = 0
            for d in rng.permutation(D):
                ids, counts = docs[d]
                old = z[d]
                m_z[old] -= 1
                n_z[old] -= counts.sum()
                n_zw[old, ids] -= counts

                new = rng.choice(K, p=self._softmax(self._log_prob(docs[d], m_z, n_z, n_zw, D)))
                z[d] = new
                m_z[new] += 1
                n_z[new] += counts.sum()
                n_zw[new, ids] += counts
                moved += new != old
            if moved == 0:  # все документы остались на местах — сошлось
                break

        # Оставляем непустые темы и нумеруем их подряд.
        alive = np.where(m_z > 0)[0]
        self.m_z_, self.n_z_, self.n_zw_, self.n_docs_ = m_z[alive], n_z[alive], n_zw[alive], D
        doc_topic = self._doc_topic(docs)
        labels = np.searchsorted(alive, z)
        topic_word = (self.n_zw_ + self.beta) / (self.n_z_[:, None] + V * self.beta)
        # Вес слова в теме делим на его среднюю частоту по всем темам, чтобы наверх
        # поднимались характерные слова, а не просто самые частые.
        distinct = topic_word * np.log(topic_word / topic_word.mean(axis=0, keepdims=True) + 1)
        return TopicResult(labels, top_words(distinct, self.vocab_), doc_topic)

    def _doc_topic(self, docs) -> np.ndarray:
        return np.vstack([
            self._softmax(self._log_prob(doc, self.m_z_, self.n_z_, self.n_zw_, self.n_docs_ + 1)) for doc in docs
        ])

    def predict(self, texts, tokens) -> np.ndarray:
        return self._doc_topic(self._encode(tokens)).argmax(axis=1)
