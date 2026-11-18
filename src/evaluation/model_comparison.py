"""Сравнение тематических моделей на датасете с эталонными метками.

    python -m src.evaluation.model_comparison              # всё
    python -m src.evaluation.model_comparison --part k     # только подбор числа тем
    python -m src.evaluation.model_comparison --part models --encoders intfloat/multilingual-e5-large

Часть «k»: для каждого числа тем считаем связность, разнообразие и ARI —
чтобы выбрать число тем так, как это делают без меток (по связности), и
проверить, совпадает ли выбор с правдой.

Часть «models»: все модели при одном числе тем, полный набор метрик,
включая устойчивость. Результаты сохраняются в docs/results/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import coherence, external_metrics, random_baseline, stability, topic_diversity
from src.models import EmbeddingTopicModel, Encoder, GSDMMTopicModel, LDATopicModel, NMFTopicModel, TopicModel
from src.preprocessing import load_table, preprocess_dataframe

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "raw" / "exit_interviews_labeled.csv"
RESULTS = ROOT / "docs" / "results"
NOISE = "noise"

ENCODERS = [
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "deepvk/USER-bge-m3",
    "ai-forever/FRIDA",
]


def load_corpus() -> tuple[list[str], list[list[str]], np.ndarray]:
    """Очищенные тексты, леммы и эталонные метки (без пустых ответов и дублей)."""
    clean, _ = preprocess_dataframe(load_table(LABELED), "reason_text")
    clean = clean[clean["true_topic"] != NOISE].reset_index(drop=True)
    return clean["clean_text"].tolist(), clean["tokens"].tolist(), clean["true_topic"].to_numpy()


def evaluate(model: TopicModel, texts, tokens, y, with_stability: bool = True) -> dict:
    t0 = time.time()
    result = model.fit(texts, tokens)
    seconds = time.time() - t0
    kw = result.keywords(10)
    row = {
        "модель": model.name,
        "тем": result.n_topics,
        **external_metrics(y, result.labels),
        "c_v": coherence(kw, tokens, "c_v")[0],
        "NPMI": coherence(kw, tokens, "c_npmi")[0],
        "diversity": topic_diversity(kw),
        "время, с": seconds,
    }
    if with_stability:
        def fit_predict(idx, seed):
            m = _clone_with_seed(model, seed)
            m.fit([texts[i] for i in idx], [tokens[i] for i in idx])
            return m.predict(texts, tokens)
        row["stability"] = stability(fit_predict, len(texts), n_runs=5)["stability_ARI_mean"]
    return row


def _clone_with_seed(model: TopicModel, seed: int) -> TopicModel:
    import copy

    m = copy.copy(model)
    m.seed = seed
    return m


def models_at(k: int, encoders: list[str]) -> list[Callable[[], TopicModel]]:
    factories: list[Callable[[], TopicModel]] = [
        lambda: LDATopicModel(k),
        lambda: LDATopicModel(k, doc_topic_prior=0.05, topic_word_prior=0.05),
        lambda: NMFTopicModel(k),
        lambda: GSDMMTopicModel(2 * k),  # максимум тем с запасом, итоговое число модель найдёт сама
    ]
    for name in encoders:
        enc = Encoder(name)
        factories += [
            lambda enc=enc: EmbeddingTopicModel(k, encoder=enc, clustering="kmeans"),
            lambda enc=enc: EmbeddingTopicModel(k, encoder=enc, clustering="agglomerative"),
            lambda enc=enc: EmbeddingTopicModel(k, encoder=enc, clustering="hdbscan", umap_dims=5, min_cluster_size=10),
        ]
    return factories


def part_k(texts, tokens, y, encoder: str, ks=range(4, 17)) -> pd.DataFrame:
    rows = []
    enc = Encoder(encoder)
    for k in ks:
        for model in (LDATopicModel(k, doc_topic_prior=0.05, topic_word_prior=0.05), NMFTopicModel(k),
                      EmbeddingTopicModel(k, encoder=enc)):
            row = evaluate(model, texts, tokens, y, with_stability=False)
            row["k"] = k
            rows.append(row)
            print(f"k={k:2d} {row['модель']:40s} c_v={row['c_v']:.3f} ARI={row['ARI']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_models(texts, tokens, y, k: int, encoders: list[str]) -> pd.DataFrame:
    rows = []
    for make in models_at(k, encoders):
        model = make()
        row = evaluate(model, texts, tokens, y)
        if isinstance(model, LDATopicModel) and model.doc_topic_prior is not None:
            row["модель"] = "LDA (α=β=0.05)"
        rows.append(row)
        print(f"{row['модель']:45s} ARI={row['ARI']:.3f} F1={row['macro_F1']:.3f} "
              f"c_v={row['c_v']:.3f} stab={row['stability']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["k", "models", "all"], default="all")
    parser.add_argument("--k", type=int, default=10, help="число тем для сравнения моделей")
    parser.add_argument("--encoders", nargs="*", default=ENCODERS)
    args = parser.parse_args()

    texts, tokens, y = load_corpus()
    print(f"Ответов: {len(texts)}, эталонных причин: {len(set(y))}")
    print("Случайное разбиение:", {k: round(v, 3) for k, v in random_baseline(y, args.k).items()})
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.part in ("k", "all"):
        df = part_k(texts, tokens, y, args.encoders[0])
        df.round(4).to_csv(RESULTS / "n_topics.csv", index=False)
    if args.part in ("models", "all"):
        df = part_models(texts, tokens, y, args.k, args.encoders)
        df.round(4).to_csv(RESULTS / f"models_k{args.k}.csv", index=False)
        print()
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
