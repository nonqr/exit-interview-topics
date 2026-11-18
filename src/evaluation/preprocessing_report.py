"""Сколько даёт каждый шаг предобработки: сравнение вариантов на эталонных метках.

    python -m src.evaluation.preprocessing_report

1. Качество фильтра пустых ответов и дублей (precision / recall по меткам).
2. Для каждого варианта предобработки обучает NMF и LDA на одних и тех же
   ответах и считает метрики из src/evaluation/metrics.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.evaluation.metrics import coherence, external_metrics, random_baseline, stability, topic_diversity
from src.preprocessing import TextPreprocessor, load_table, preprocess_dataframe

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "raw" / "exit_interviews_labeled.csv"
N_TOPICS = 10
NOISE = "noise"

VARIANTS = {
    "1. только лемматизация": TextPreprocessor(pos_filter=False, use_domain_stopwords=False, extra_stopwords=()),
    "2. + фильтр частей речи": TextPreprocessor(pos_filter=True, use_domain_stopwords=False),
    "3. + доменные стоп-слова": TextPreprocessor(pos_filter=True, use_domain_stopwords=True),
    "4. + склейка отрицаний": TextPreprocessor(pos_filter=True, use_domain_stopwords=True, join_negations=True),
}


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Насколько точно отсекаются бессмысленные ответы."""
    tp = TextPreprocessor()
    predicted_noise = ~df["reason_text"].map(tp.is_meaningful)
    true_noise = df["true_topic"] == NOISE
    tp_ = int((predicted_noise & true_noise).sum())
    return pd.DataFrame([{
        "найдено пустых": int(predicted_noise.sum()),
        "из них верно": tp_,
        "precision": tp_ / max(int(predicted_noise.sum()), 1),
        "recall": tp_ / max(int(true_noise.sum()), 1),
        "пропущены": df.loc[true_noise & ~predicted_noise, "reason_text"].tolist(),
        "отброшены зря": df.loc[predicted_noise & ~true_noise, "reason_text"].tolist(),
    }])


def _fit(model: str, docs: list[str], idx: np.ndarray, seed: int):
    if model == "NMF":
        vec = TfidfVectorizer(min_df=2)
        X = vec.fit_transform([docs[i] for i in idx])
        mdl = NMF(N_TOPICS, random_state=seed, init="nndsvda", max_iter=500).fit(X)
    else:
        vec = CountVectorizer(min_df=2)
        X = vec.fit_transform([docs[i] for i in idx])
        mdl = LatentDirichletAllocation(N_TOPICS, random_state=seed, learning_method="batch", max_iter=50).fit(X)
    labels = mdl.transform(vec.transform(docs)).argmax(1)
    vocab = np.array(vec.get_feature_names_out())
    topics = [list(vocab[c.argsort()[::-1][:10]]) for c in mdl.components_]
    return labels, topics


def compare_variants(clean: pd.DataFrame) -> pd.DataFrame:
    y = clean["true_topic"].to_numpy()
    rows = []
    for name, tp in VARIANTS.items():
        tokens = clean["reason_text"].map(tp.tokens).tolist()
        docs = [" ".join(t) for t in tokens]
        for model in ("NMF", "LDA"):
            labels, topics = _fit(model, docs, np.arange(len(docs)), seed=0)
            st = stability(lambda idx, s: _fit(model, docs, idx, s)[0], len(docs), n_runs=5)
            rows.append({
                "вариант": name,
                "модель": model,
                **{k: round(v, 3) for k, v in external_metrics(y, labels).items()},
                "c_v": round(coherence(topics, tokens, "c_v")[0], 3),
                "diversity": round(topic_diversity(topics), 2),
                "stability": round(st["stability_ARI_mean"], 3),
                "словарь": len({w for t in tokens for w in t}),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_table(LABELED)
    print("== Фильтр пустых ответов ==")
    fq = filter_quality(df)
    print(fq.drop(columns=["пропущены", "отброшены зря"]).round(3).to_string(index=False))
    print("Пропущены:", fq.loc[0, "пропущены"])
    print("Отброшены зря:", fq.loc[0, "отброшены зря"])

    clean, report = preprocess_dataframe(df, "reason_text")
    print(f"\nДублей удалено: {report.n_duplicates}")
    clean = clean[clean["true_topic"] != NOISE].reset_index(drop=True)  # пропущенный мусор не оцениваем
    print(f"Ответов для оценки тем: {len(clean)}\n")

    print("== Случайное разбиение ==")
    print({k: round(v, 3) for k, v in random_baseline(clean["true_topic"], N_TOPICS).items()})
    print("\n== Варианты предобработки ==")
    pd.set_option("display.width", 200)
    print(compare_variants(clean).to_string(index=False))


if __name__ == "__main__":
    main()
