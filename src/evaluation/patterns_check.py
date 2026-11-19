"""Видит ли сервис закономерности, заложенные в датасет.

    python -m src.evaluation.patterns_check [--mode "Быстрый (e5-large-instruct)"]

В синтетический датасет специально заложены пять закономерностей (docs/dataset.md).
Скрипт прогоняет полный анализ, как в веб-интерфейсе, и проверяет каждую
закономерность дважды:
  * по эталонным причинам — что она действительно есть в данных;
  * по темам, найденным моделью, — что её увидит HR в дашборде.
Чтобы сравнить темы модели с причинами, каждой теме сопоставляется причина,
которая в ней встречается чаще всего (так тему назвал бы человек, прочитав ответы).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.app.analysis import DEFAULT_MODE, MODES, TOPIC_COL, run_analysis
from src.evaluation.metrics import majority_mapping
from src.preprocessing import load_table

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "raw" / "exit_interviews_labeled.csv"


def lift(df: pd.DataFrame, col: str, mask: pd.Series, reason: str) -> float:
    """Во сколько раз причина чаще встречается в группе mask, чем во всём датасете."""
    overall = (df[col] == reason).mean()
    return float((df.loc[mask, col] == reason).mean() / overall) if overall else 0.0


def check(df: pd.DataFrame, col: str) -> list[dict]:
    dates = pd.to_datetime(df["exit_date"])
    after = dates >= "2025-03-01"
    sat = df.groupby(col)["satisfaction"].apply(lambda s: s.astype(float).mean())
    sat = sat.drop("noise", errors="ignore")
    share_before = (df.loc[~after, col] == "schedule").mean()
    share_after = (df.loc[after, col] == "schedule").mean()
    checks = [
        ("IT чаще уходит из-за переработок",
         lift(df, col, df["department"] == "IT", "workload"), 1.5),
        ("Склад и производство — из-за условий труда",
         lift(df, col, df["department"].isin(["Логистика и склад", "Производство"]), "conditions"), 1.5),
        ("Новички (до 6 мес.) — из-за неинтересных задач",
         lift(df, col, df["tenure_group"] == "<6 мес", "content"), 1.3),
        ("С марта 2025 растёт доля ухода из-за графика (отмена удалёнки)",
         share_after / share_before if share_before else 0.0, 1.5),
    ]
    rows = [{"закономерность": name, "значение": round(v, 2), "порог": thr, "видна": v >= thr}
            for name, v, thr in checks]
    rows.append({"закономерность": "Самая низкая оценка компании — у конфликта с руководителем, самая высокая — у личных причин",
                 "значение": f"{sat.idxmin()} / {sat.idxmax()}", "порог": "management / personal",
                 "видна": sat.idxmin() == "management" and sat.idxmax() == "personal"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=list(MODES))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    df = load_table(LABELED)
    analysis = run_analysis(df, "reason_text", args.mode, args.k)
    data = analysis.data
    mapping = majority_mapping(data["true_topic"], data[TOPIC_COL])
    data["predicted_reason"] = data[TOPIC_COL].map(mapping)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(f"Режим: {args.mode}, тем: {analysis.result.n_topics}\n")
    print("Темы модели → причина, которая в теме встречается чаще всего:")
    for t, name in analysis.names.items():
        print(f"  {name:45s} → {mapping.get(t)}")
    for title, col in (("По эталонным причинам", "true_topic"), ("По темам модели (то, что увидит HR)", "predicted_reason")):
        print(f"\n== {title} ==")
        print(pd.DataFrame(check(data, col)).to_string(index=False))


if __name__ == "__main__":
    main()
