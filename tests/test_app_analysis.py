import numpy as np
import pandas as pd
import pytest

from src.app.analysis import (
    CONF_COL, NAME_COL, TOPIC_COL, apply_names, date_columns, default_names, lift_by_segment, run_analysis,
    segment_columns, share_by_segment, suggest_n_topics, timeline, to_excel,
)
from src.models import EmbeddingTopicModel

PHRASES = {
    "salary": ["мало платят", "низкая зарплата", "зарплату не поднимали", "маленький оклад и нет премий"],
    "boss": ["начальник хамит", "руководитель кричит на всех", "конфликт с руководителем", "плохой начальник"],
    "move": ["переезжаю в другой город", "переезд к семье", "уезжаю в другой регион", "семья переезжает"],
}


class FakeEncoder:
    model_name = "fake/encoder"

    def encode(self, texts):
        keys = {"salary": ["плат", "зарплат", "оклад"], "boss": ["начальник", "руковод"], "move": ["пере", "уезж"]}
        out = []
        for t in texts:
            v = np.array([sum(k in t.lower() for k in ks) for ks in keys.values()], dtype=float) + 0.01
            v = v + np.random.default_rng(len(t)).normal(0, 0.02, 3)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


@pytest.fixture(scope="module")
def df():
    rows = []
    for i in range(60):
        label = list(PHRASES)[i % 3]
        rows.append({
            "id": str(i), "Отдел": ["IT", "Склад"][int(label == "salary")] if i % 4 else "HR",
            "Дата": f"2025-{(i % 12) + 1:02d}-15", "Ответ": PHRASES[label][i % 4] + ("" if i < 3 else f" {i}"),
        })
    rows.append({"id": "99", "Отдел": "IT", "Дата": "2025-01-01", "Ответ": "нет"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def analysis(df):
    model = EmbeddingTopicModel(n_topics=3, encoder=FakeEncoder())
    return run_analysis(df, "Ответ", n_topics=3, model=model)


def test_run_analysis(analysis):
    assert analysis.report.n_empty == 1  # «нет» отброшен
    assert len(analysis.data) == 60
    assert analysis.data[TOPIC_COL].nunique() == 3
    assert analysis.data[CONF_COL].between(0, 1).all()
    assert analysis.topic_table()["Ответов"].sum() == 60


def test_default_names_are_unique_keywords(analysis):
    names = default_names(analysis.result)
    assert len(set(names.values())) == len(names)


def test_rename_and_merge(analysis):
    names = dict(analysis.names)
    apply_names(analysis, {0: "Деньги", 1: "Деньги", 2: names[2]})
    assert set(analysis.data[NAME_COL]) == {"Деньги", names[2]}
    assert len(analysis.topic_table()) == 2
    apply_names(analysis, names)  # вернуть как было


def test_segments_and_dates(analysis):
    assert "Отдел" in segment_columns(analysis.data, exclude=("Ответ",))
    assert "Ответ" not in segment_columns(analysis.data, exclude=("Ответ",))
    assert date_columns(analysis.data, exclude=("Ответ",)) == ["Дата"]
    share = share_by_segment(analysis, "Отдел")
    assert np.allclose(share.sum(axis=1), 100)
    lift = lift_by_segment(analysis, "Отдел")
    assert {"Сегмент", "Тема", "Подъём", "z"} <= set(lift.columns)
    assert lift["z"].is_monotonic_decreasing
    tl = timeline(analysis, "Дата")
    assert np.allclose(tl.sum(axis=1), 100)


def test_suggest_n_topics_finds_clear_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:6]
    X = np.vstack([c + rng.normal(0, 0.05, 8) for c in centers for _ in range(20)])
    k, scores = suggest_n_topics(X, range(3, 10))
    assert k == 6 and set(scores) == set(range(3, 10))


def test_excel_export(analysis):
    data = to_excel(analysis)
    sheets = pd.read_excel(__import__("io").BytesIO(data), sheet_name=None)
    assert set(sheets) == {"Ответы", "Темы", "Отчёт"}
    assert len(sheets["Ответы"]) == 60


def test_too_few_answers():
    tiny = pd.DataFrame({"Ответ": ["мало платят", "нет"]})
    with pytest.raises(ValueError, match="слишком мало"):
        run_analysis(tiny, "Ответ", mode="Без нейросети (NMF)")


def test_notable_ignores_tiny_segments():
    from src.app.analysis import notable

    insights = pd.DataFrame([
        {"Сегмент": "HR", "Тема": "A", "Ответов": 2, "Размер сегмента": 3, "Подъём": 4.0, "z": 1.5},
        {"Сегмент": "IT", "Тема": "B", "Ответов": 40, "Размер сегмента": 80, "Подъём": 2.0, "z": 4.0},
    ])
    assert notable(insights)["Сегмент"].tolist() == ["IT"]
