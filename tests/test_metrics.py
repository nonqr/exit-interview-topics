import numpy as np
import pytest

from src.evaluation.metrics import (
    annotator_agreement,
    coherence,
    external_metrics,
    hungarian_mapping,
    majority_mapping,
    per_label_report,
    random_baseline,
    stability,
    topic_diversity,
    word_intrusion_tasks,
)

TRUE = ["salary"] * 4 + ["team"] * 4 + ["career"] * 4


def test_perfect_clustering_scores_one():
    pred = [0] * 4 + [1] * 4 + [2] * 4
    m = external_metrics(TRUE, pred)
    assert m["ARI"] == pytest.approx(1.0)
    assert m["NMI"] == pytest.approx(1.0)
    assert m["purity"] == pytest.approx(1.0)
    assert m["macro_F1"] == pytest.approx(1.0)


def test_metrics_do_not_depend_on_topic_numbering():
    pred = [2] * 4 + [0] * 4 + [1] * 4
    assert external_metrics(TRUE, pred)["ARI"] == pytest.approx(1.0)


def test_random_baseline_is_near_zero():
    true = np.repeat(np.arange(10), 50)
    assert abs(random_baseline(true, 10)["ARI"]) < 0.02


def test_mappings():
    pred = [0, 0, 0, 1] + [1] * 4 + [2] * 4
    assert majority_mapping(TRUE, pred) == {0: "salary", 1: "team", 2: "career"}
    assert hungarian_mapping(TRUE, pred) == {0: "salary", 1: "team", 2: "career"}


def test_per_label_report_counts_errors():
    pred = [0, 0, 0, 1] + [1] * 4 + [2] * 4  # один «salary» ушёл в тему «team»
    report = per_label_report(TRUE, pred)
    assert report.loc["salary", "recall"] == pytest.approx(0.75)
    assert report.loc["team", "precision"] == pytest.approx(0.8)


def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topic_diversity([["a", "b"], ["a", "b"]]) == 0.5


def test_coherent_topics_score_higher():
    docs = [["зарплата", "низкий", "оклад"]] * 10 + [["коллектив", "токсичный", "сплетня"]] * 10
    good, _ = coherence([["зарплата", "оклад", "низкий"], ["коллектив", "сплетня", "токсичный"]], docs, "c_npmi")
    bad, _ = coherence([["зарплата", "сплетня", "низкий"], ["коллектив", "оклад", "токсичный"]], docs, "c_npmi")
    assert good > bad


def test_stability_of_deterministic_model_is_one():
    labels = np.repeat(np.arange(3), 10)
    result = stability(lambda idx, seed: labels, n_docs=30, n_runs=3)
    assert result["stability_ARI_mean"] == pytest.approx(1.0)


def test_word_intrusion_tasks_contain_intruder():
    tasks = word_intrusion_tasks([["a", "b", "c"], ["x", "y", "z"]], n_words=3)
    for _, row in tasks.iterrows():
        assert row["intruder"] in row["words"].split(", ")


def test_annotator_agreement():
    assert annotator_agreement(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
