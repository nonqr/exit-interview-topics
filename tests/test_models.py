import numpy as np
import pytest

from src.evaluation.metrics import external_metrics
from src.models import OUTLIER, EmbeddingTopicModel, GSDMMTopicModel, LDATopicModel, NMFTopicModel, TopicResult
from src.models.embedding import class_tfidf

# Игрушечный корпус: три явные темы по 20 документов.
VOCAB = {
    "salary": ["зарплата", "оклад", "платить", "премия", "деньги"],
    "boss": ["руководитель", "начальник", "хамить", "кричать", "контроль"],
    "move": ["переезд", "город", "семья", "переехать", "регион"],
}
rng = np.random.default_rng(0)
TOKENS, LABELS = [], []
for label, words in VOCAB.items():
    for _ in range(20):
        TOKENS.append(list(rng.choice(words, size=4)))
        LABELS.append(label)
TEXTS = [" ".join(t) for t in TOKENS]


class FakeEncoder:
    """Вместо e5: вектор = индикатор темы + шум. Не требует скачивания модели."""

    model_name = "fake/encoder"

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.array([sum(w in t for w in words) for words in VOCAB.values()], dtype=float)
            v = v + np.random.default_rng(len(t)).normal(0, 0.05, 3)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


@pytest.mark.parametrize("model", [
    NMFTopicModel(n_topics=3),
    LDATopicModel(n_topics=3, doc_topic_prior=0.1),
    EmbeddingTopicModel(n_topics=3, encoder=FakeEncoder()),
    EmbeddingTopicModel(n_topics=3, clustering="agglomerative", encoder=FakeEncoder()),
])
def test_models_recover_obvious_topics(model):
    result = model.fit(TEXTS, TOKENS)
    assert isinstance(result, TopicResult)
    assert result.n_topics == 3
    assert external_metrics(LABELS, result.labels)["ARI"] > 0.8
    # Ключевые слова каждой темы должны быть из одного словаря.
    for kw in result.keywords(3):
        assert any(set(kw) <= set(words) for words in VOCAB.values())


@pytest.mark.parametrize("model", [NMFTopicModel(n_topics=3), GSDMMTopicModel(n_topics=3),
                                   EmbeddingTopicModel(n_topics=3, encoder=FakeEncoder())])
def test_predict_matches_fit_labels(model):
    labels = model.fit(TEXTS, TOKENS).labels
    assert (model.predict(TEXTS, TOKENS) == labels).mean() > 0.9


def test_hdbscan_finds_topics_without_n_topics():
    result = EmbeddingTopicModel(clustering="hdbscan", min_cluster_size=5, encoder=FakeEncoder()).fit(TEXTS, TOKENS)
    # HDBSCAN сам выбирает число тем и может разделить тему на части — проверяем совпадение с эталоном.
    assert result.n_topics >= 3
    assert external_metrics(LABELS, result.labels)["NMI"] > 0.8
    assert OUTLIER not in result.labels  # выбросы переназначены ближайшей теме


def test_topic_result_helpers():
    result = NMFTopicModel(n_topics=3).fit(TEXTS, TOKENS)
    info = result.topic_info()
    assert info["size"].sum() == len(TEXTS)
    assert np.all((result.confidence() > 0) & (result.confidence() <= 1))
    docs = result.representative_docs(TEXTS, topic=0, n=3)
    assert len(docs) == 3 and all(d in TEXTS for d in docs)


def test_class_tfidf_prefers_distinctive_words():
    tokens = [["зарплата", "общий"], ["зарплата", "общий"], ["переезд", "общий"], ["переезд", "общий"]]
    weights, vocab = class_tfidf(tokens, np.array([0, 0, 1, 1]), 2, min_df=1)
    vocab = list(vocab)
    assert weights[0, vocab.index("зарплата")] > weights[0, vocab.index("общий")]
    assert weights[1, vocab.index("переезд")] > weights[1, vocab.index("общий")]


def test_gsdmm_empties_extra_topics():
    # GSDMM задают максимум тем с запасом: при максимуме, равном настоящему числу,
    # две темы могут слиться. Здесь максимум 10, а настоящих 3 — лишние должны опустеть.
    result = GSDMMTopicModel(n_topics=10, seed=1).fit(TEXTS, TOKENS)
    assert result.n_topics < 10
    assert external_metrics(LABELS, result.labels)["NMI"] > 0.8


def test_encoder_cache_depends_on_prefix():
    # С другой инструкцией модель выдаёт другие векторы — кэш не должен их путать.
    from src.models.embedding import Encoder

    a, b = Encoder("intfloat/multilingual-e5-large-instruct"), Encoder("intfloat/multilingual-e5-large-instruct")
    b.prefix = "Instruct: другая инструкция\nQuery: "
    assert a._cache_path != b._cache_path
