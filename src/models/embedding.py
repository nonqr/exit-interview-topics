"""Тематическая модель на эмбеддингах (подход BERTopic).

1. Каждый ответ превращаем в вектор смысла моделью multilingual-e5 (или другой).
   Похожие по смыслу ответы получают близкие векторы, даже если в них нет общих
   слов: «мало платят» и «низкий оклад».
2. Кластеризуем векторы: KMeans / агломеративная кластеризация с заданным числом
   тем или HDBSCAN, который сам находит число тем и отделяет выбросы.
3. Ключевые слова темы считаем через c-TF-IDF по леммам: склеиваем все ответы
   темы в один «документ» и ищем слова, частые в этой теме и редкие в остальных.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer

from src.models.base import OUTLIER, TopicModel, TopicResult, top_words

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "models_cache" / "embeddings"

# Лучшая модель по итогам сравнения (docs/modeling.md). Понимает инструкцию — ей можно
# сказать, что в ответе важна причина увольнения. Весит ~8 ГБ (bfloat16).
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
# Лёгкая альтернатива (~2 ГБ), если не хватает памяти: качество ниже (ARI 0.42 против 0.68).
LIGHT_MODEL = "intfloat/multilingual-e5-large-instruct"

# Многие модели обучены с префиксом-инструкцией. Без него качество заметно падает.
# e5: для кластеризации и классификации авторы рекомендуют «query: ».
# FRIDA: префикс задачи «categorize_topic: » для тематической кластеризации.
PREFIXES = {
    "intfloat/multilingual-e5-large": "query: ",
    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/multilingual-e5-small": "query: ",
    # Инструктивная версия e5: инструкция говорит модели, на что обращать внимание.
    # Без неё ответы группируются в том числе по форме (короткие, эмоциональные),
    # с ней — по причине увольнения. На нашем датасете ARI вырос с 0.21 до 0.43.
    "intfloat/multilingual-e5-large-instruct":
        "Instruct: Identify the main reason why the employee is leaving the company\nQuery: ",
    "ai-forever/FRIDA": "categorize_topic: ",
    # Модели на основе LLM с инструкциями (формат из их документации).
    "ai-sage/Giga-Embeddings-instruct":
        "Instruct: Identify the main reason why the employee is leaving the company\nQuery: ",
    "Qwen/Qwen3-Embedding-0.6B":
        "Instruct: Identify the main reason why the employee is leaving the company\nQuery:",
    "Qwen/Qwen3-Embedding-4B":
        "Instruct: Identify the main reason why the employee is leaving the company\nQuery:",
    "deepvk/USER-bge-m3": "",
    "BAAI/bge-m3": "",
}


# Дополнительные параметры загрузки для больших моделей. bfloat16 вдвое уменьшает
# потребление памяти. Giga-Embeddings использует собственный код модели из своего
# репозитория (trust_remote_code); код проверен: это только архитектура сети.
LOAD_KWARGS: dict[str, dict] = {
    "ai-sage/Giga-Embeddings-instruct": {
        "trust_remote_code": True,
        "model_kwargs": {"dtype": "bfloat16", "attn_implementation": "eager", "trust_remote_code": True},
        "config_kwargs": {"trust_remote_code": True},
    },
    "Qwen/Qwen3-Embedding-4B": {"model_kwargs": {"dtype": "bfloat16"}},
}


class Encoder:
    """Обёртка над sentence-transformers с кэшем на диске.

    Подсчёт эмбеддингов e5-large на CPU занимает около минуты на 500 ответов,
    поэтому векторы сохраняются в models_cache/ по каждому тексту отдельно:
    при повторном запуске и на подвыборках считаются только новые тексты.
    """

    _loaded: dict[str, object] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, use_cache: bool = True):
        self.model_name, self.batch_size, self.use_cache = model_name, batch_size, use_cache
        self.prefix = PREFIXES.get(model_name, "")
        self._memory: dict[str, np.ndarray] | None = None

    def _model(self):
        if self.model_name not in Encoder._loaded:
            from sentence_transformers import SentenceTransformer  # тяжёлый импорт — только когда нужен

            Encoder._loaded[self.model_name] = SentenceTransformer(
                self.model_name, device="cpu", **LOAD_KWARGS.get(self.model_name, {})
            )
        return Encoder._loaded[self.model_name]

    @property
    def _cache_path(self) -> Path:
        # Префикс входит в ключ: с другой инструкцией получаются другие векторы.
        tag = hashlib.sha1(self.prefix.encode("utf-8")).hexdigest()[:8]
        return CACHE_DIR / f"{self.model_name.replace('/', '__')}__{tag}.npz"

    def _cache(self) -> dict[str, np.ndarray]:
        if self._memory is None:
            self._memory = {}
            if self.use_cache and self._cache_path.exists():
                data = np.load(self._cache_path, allow_pickle=False)
                self._memory = dict(zip(data["texts"].tolist(), data["emb"]))
        return self._memory

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        cache = self._cache()
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            emb = self._model().encode(
                [self.prefix + t for t in missing], batch_size=self.batch_size,
                normalize_embeddings=True, show_progress_bar=len(missing) > 200,
            )
            cache.update(zip(missing, np.asarray(emb, dtype=np.float32)))
            if self.use_cache:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                keys = list(cache)
                np.savez(self._cache_path, texts=np.array(keys), emb=np.vstack([cache[k] for k in keys]))
        return np.vstack([cache[t] for t in texts])


def class_tfidf(tokens: Sequence[Sequence[str]], labels: np.ndarray, n_topics: int, min_df: int = 2):
    """c-TF-IDF из BERTopic: веса слов для каждой темы.

    tf(t, c)  — доля слова t среди всех слов темы c;
    idf(t)    — log(1 + A / f(t)), где A — среднее число слов в теме,
                f(t) — сколько раз слово встречается во всех темах.
    Слово получает большой вес, если оно частое в своей теме и редкое в остальных.
    """
    docs = [" ".join(t) for t in tokens]
    vec = CountVectorizer(analyzer=str.split, min_df=min_df)
    X = vec.fit_transform(docs)
    per_topic = np.vstack([np.asarray(X[labels == t].sum(axis=0)).ravel() for t in range(n_topics)])
    tf = per_topic / np.maximum(per_topic.sum(axis=1, keepdims=True), 1)
    avg_words = per_topic.sum() / n_topics
    idf = np.log(1 + avg_words / np.maximum(per_topic.sum(axis=0), 1))
    return tf * idf, vec.get_feature_names_out()


class EmbeddingTopicModel(TopicModel):
    def __init__(
        self,
        n_topics: int = 10,
        model_name: str = DEFAULT_MODEL,
        clustering: str = "kmeans",
        umap_dims: int | None = None,
        min_cluster_size: int = 10,
        reassign_outliers: bool = True,
        temperature: float = 0.05,
        seed: int = 0,
        encoder: Encoder | None = None,
    ):
        """
        clustering: "kmeans" | "agglomerative" | "hdbscan".
        umap_dims:  снизить размерность через UMAP перед кластеризацией (None — без UMAP).
                    BERTopic делает так по умолчанию; для HDBSCAN это почти обязательно.
        min_cluster_size: минимальный размер темы для HDBSCAN.
        reassign_outliers: отнести выбросы HDBSCAN к ближайшей теме.
        temperature: «резкость» уверенности: чем меньше, тем ближе уверенность к 0 или 1.
        """
        self.n_topics, self.clustering, self.umap_dims = n_topics, clustering, umap_dims
        self.min_cluster_size, self.reassign_outliers = min_cluster_size, reassign_outliers
        self.temperature, self.seed = temperature, seed
        self.encoder = encoder or Encoder(model_name)
        short = self.encoder.model_name.split("/")[-1]
        self.name = f"{short} + {clustering}" + (f" + UMAP{umap_dims}" if umap_dims else "")
        self.embeddings_: np.ndarray | None = None

    def _reduce(self, emb: np.ndarray) -> np.ndarray:
        if not self.umap_dims:
            return emb
        import umap  # тяжёлый импорт

        return umap.UMAP(n_components=self.umap_dims, n_neighbors=15, min_dist=0.0, metric="cosine",
                         random_state=self.seed).fit_transform(emb)

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        if self.clustering == "kmeans":
            return KMeans(self.n_topics, n_init=10, random_state=self.seed).fit_predict(X)
        if self.clustering == "agglomerative":
            return AgglomerativeClustering(self.n_topics, linkage="ward").fit_predict(X)
        if self.clustering == "hdbscan":
            return HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=5, copy=True).fit_predict(X)
        raise ValueError(f"Неизвестный метод кластеризации: {self.clustering}")

    def fit(self, texts, tokens) -> TopicResult:
        emb = self.encoder.encode(list(texts))
        self.embeddings_ = emb
        labels = self._cluster(self._reduce(emb))

        # Перенумеровываем темы подряд (у HDBSCAN номера могут идти с пропусками).
        topics = sorted(set(labels) - {OUTLIER})
        remap = {t: i for i, t in enumerate(topics)}
        labels = np.array([remap.get(t, OUTLIER) for t in labels])
        n = len(topics)

        # Центр темы — средний вектор её ответов. Уверенность — softmax близости к центрам.
        centroids = np.vstack([emb[labels == t].mean(axis=0) for t in range(n)])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        sim = emb @ centroids.T
        z = (sim - sim.max(axis=1, keepdims=True)) / self.temperature
        doc_topic = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

        if self.reassign_outliers:
            labels = np.where(labels == OUTLIER, sim.argmax(axis=1), labels)
        self.centroids_ = centroids

        weights, vocab = class_tfidf(tokens, labels, n)
        return TopicResult(labels, top_words(weights, vocab), doc_topic)

    def predict(self, texts, tokens) -> np.ndarray:
        """Новый ответ относится к теме с ближайшим центром."""
        return (self.encoder.encode(list(texts)) @ self.centroids_.T).argmax(axis=1)
