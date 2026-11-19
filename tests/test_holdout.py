"""Отложенный тестовый набор должен оставаться независимым от основного."""

import pandas as pd

from src.generation.build_dataset import HOLDOUT_SEED, HOLDOUT_SOURCE_DIR, SOURCE_DIR, build, read_source
from src.preprocessing import TextPreprocessor


def test_holdout_texts_do_not_repeat_main_dataset():
    tp = TextPreprocessor()
    main = [set(tp.tokens(r["reason_text"])) for r in read_source(SOURCE_DIR)]
    for rec in read_source(HOLDOUT_SOURCE_DIR):
        words = set(tp.tokens(rec["reason_text"]))
        if len(words) < 3:  # короткие ответы («нет», «скучно») совпадают естественно
            continue
        closest = max(len(words & m) / len(words | m) for m in main)
        assert closest < 0.6, f"слишком похож на ответ из основного набора: {rec['reason_text']}"


def test_holdout_build_is_reproducible():
    a = build(HOLDOUT_SOURCE_DIR, HOLDOUT_SEED, n_duplicates=3, id_prefix="HO")
    b = build(HOLDOUT_SOURCE_DIR, HOLDOUT_SEED, n_duplicates=3, id_prefix="HO")
    pd.testing.assert_frame_equal(a, b)
    assert set(a["true_topic"]) == set(pd.read_csv("data/holdout/exit_interviews_holdout_labeled.csv")["true_topic"])
