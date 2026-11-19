"""Проверка на отложенном тестовом наборе: держится ли качество на новых данных.

    python -m src.evaluation.holdout_check

Все параметры (стоп-слова, модель, число тем, инструкция для эмбеддингов) подбирались
на основном датасете. Если качество измерять на нём же, оно может быть завышено:
мы невольно подстроились под конкретные формулировки. Поэтому отдельно написан
отложенный набор (data/holdout_source/, ~200 ответов): другие формулировки, другой
стиль (разговорный, сленг, опечатки), другие пропорции причин. При настройке он не
использовался.

Два сценария:
  A. Модель обучается заново на новом наборе — как если бы HR загрузил новую выгрузку.
  B. Модель, обученная на основном датасете, относит новые ответы к уже найденным темам
     (ближайший центр темы) — как работал бы сервис, если темы зафиксированы, а выгрузки
     приходят каждый месяц. Каждой теме сопоставлена причина, которая встречается в ней
     чаще всего на основном датасете; на новом наборе проверяется, угадана ли причина.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.analysis import MODES, TOPIC_COL, run_analysis
from src.evaluation.metrics import external_metrics, majority_mapping, per_label_report, random_baseline
from src.evaluation.patterns_check import check as check_patterns
from src.preprocessing import TextPreprocessor, load_table, preprocess_dataframe

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "data" / "raw" / "exit_interviews_labeled.csv"
HOLDOUT = ROOT / "data" / "holdout" / "exit_interviews_holdout_labeled.csv"
RESULTS = ROOT / "docs" / "results" / "holdout.csv"
NOISE = "noise"


def quality(data: pd.DataFrame, pred_col: str = TOPIC_COL) -> dict[str, float]:
    """Метрики по ответам с настоящей причиной (без «мусора», как в сравнении моделей)."""
    d = data[data["true_topic"] != NOISE]
    return {k: round(v, 3) for k, v in external_metrics(d["true_topic"], d[pred_col]).items()}


def cleaning(df: pd.DataFrame, data: pd.DataFrame) -> dict[str, float]:
    """Насколько очистка отсеивает ответы без содержания и не задевает настоящие."""
    total_noise = df.loc[df["true_topic"] == NOISE, "reason_text"].nunique()
    kept_noise = data.loc[data["true_topic"] == NOISE, "reason_text"].nunique()
    total_real = df.loc[df["true_topic"] != NOISE, "reason_text"].nunique()
    kept_real = data.loc[data["true_topic"] != NOISE, "reason_text"].nunique()
    return {"мусор отсеян": f"{total_noise - kept_noise} из {total_noise}",
            "настоящих ответов сохранено": f"{kept_real} из {total_real}"}


def transfer(main: pd.DataFrame, hold: pd.DataFrame, mode: str, k: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Сценарий B: темы с основного датасета, ответы — из отложенного."""
    analysis = run_analysis(main, "reason_text", mode, k)
    model = analysis.model
    mapping = majority_mapping(analysis.data["true_topic"], analysis.data[TOPIC_COL])

    clean, _ = preprocess_dataframe(hold, "reason_text", TextPreprocessor())
    clean = clean[clean["true_topic"] != NOISE].reset_index(drop=True)
    pred = pd.Series(model.predict(clean["clean_text"].tolist(), clean["tokens"].tolist())).map(mapping)
    clean["predicted_reason"] = pred
    report = per_label_report(clean["true_topic"], pred, mapping={r: r for r in set(mapping.values())})
    row = {"точность": round(float((pred == clean["true_topic"]).mean()), 3),
           "macro_F1": round(float(report["f1"].mean()), 3),
           "причин без своей темы": ", ".join(sorted(set(clean["true_topic"]) - set(mapping.values()))) or "нет"}
    return row, report, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    main_df, hold_df = load_table(MAIN), load_table(HOLDOUT)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"Основной набор: {len(main_df)} записей, отложенный: {len(hold_df)}\n")

    rows, reports = [], {}
    for mode in MODES:
        a_main = run_analysis(main_df, "reason_text", mode, args.k)
        a_hold = run_analysis(hold_df, "reason_text", mode, args.k)
        b_row, b_report, b_clean = transfer(main_df, hold_df, mode, args.k)
        reports[mode] = b_report
        for name, a in (("основной", a_main), ("отложенный", a_hold)):
            rows.append({"режим": mode, "набор": name, "сценарий": "A: обучение заново", **quality(a.data)})
        rows.append({"режим": mode, "набор": "отложенный", "сценарий": "B: темы с основного", **b_row})
        if mode == next(iter(MODES)):
            print("Очистка на отложенном наборе:", cleaning(hold_df, a_hold.data))
            noise_left = a_hold.data.loc[a_hold.data["true_topic"] == NOISE, "reason_text"].tolist()
            print("Прошли очистку, хотя причины в них нет:", noise_left)
            hold_clean = b_clean

    true_real = hold_clean.loc[hold_clean["true_topic"] != NOISE, "true_topic"]
    rnd = {k: round(v, 3) for k, v in random_baseline(true_real, args.k).items()}
    rows.append({"режим": "случайное разбиение", "набор": "отложенный", "сценарий": "нижняя граница", **rnd})

    table = pd.DataFrame(rows)
    print("\n== Качество ==")
    print(table.to_string(index=False))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS, index=False)

    first = next(iter(MODES))
    print(f"\n== Сценарий B, {first}: F1 по причинам ==")
    print(reports[first][["precision", "recall", "f1", "support"]].round(2).to_string())

    print(f"\n== Закономерности на отложенном наборе ({len(hold_clean)} ответов — выборка маленькая, пороги те же) ==")
    for title, col in (("по эталону", "true_topic"), (f"по темам модели (сценарий B, {first})", "predicted_reason")):
        print(f"-- {title}")
        print(pd.DataFrame(check_patterns(hold_clean, col)).to_string(index=False))
    print(f"\nТаблица сохранена: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
