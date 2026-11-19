"""Оценка людьми: понятны ли темы человеку и насколько однозначна сама задача.

    python -m src.evaluation.human_eval make                 # подготовить бланк и ключ
    python -m src.evaluation.human_eval score                # посчитать результаты по заполненным бланкам

Две части:

1. «Лишнее слово» (word intrusion, Chang et al., 2009). К пяти главным словам темы
   подмешано слово из другой темы. Если человек легко находит лишнее, тема осмысленная.
   Задания составлены по темам трёх моделей (Qwen3, e5, NMF) вперемешку, модель скрыта:
   так видно, чьи темы людям понятнее.
2. «Назовите причину». 60 ответов (по 6 на каждую причину) каждый участник независимо
   относит к одной из 10 причин. Каппа Коэна между участниками показывает, насколько задача
   однозначна для людей — это потолок, выше которого от модели требовать бессмысленно.
   Там же сравнивается согласие модели с людьми.

Порядок работы: `make` создаёт data/human_eval/human_eval_form.xlsx. Каждый участник
заполняет свою копию, не советуясь с остальными, и сохраняет её в data/human_eval/answers/
под своим именем (например, answers/stanislav.xlsx). Затем `score`.
Ключ с правильными ответами лежит в data/human_eval/key.csv — до заполнения его не открывать.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.analysis import MODES, TOPIC_COL, run_analysis
from src.evaluation.metrics import annotator_agreement, majority_mapping, word_intrusion_tasks
from src.generation.taxonomy import TOPICS
from src.preprocessing import load_table

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "raw" / "exit_interviews_labeled.csv"
OUT_DIR = ROOT / "data" / "human_eval"
FORM = OUT_DIR / "human_eval_form.xlsx"
KEY = OUT_DIR / "key.csv"
ANSWERS_DIR = OUT_DIR / "answers"
RESULTS = ROOT / "docs" / "results" / "human_eval.csv"

REASONS = {t.code: t.name for t in TOPICS}
SHEET_WORDS, SHEET_REASONS = "1. Лишнее слово", "2. Причина ухода"
PER_REASON = 6
SEED = 7


# --- подготовка бланка -----------------------------------------------------------


def intrusion_tasks(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """По 10 заданий от каждой модели, вперемешку, с номерами W01…"""
    parts = []
    for i, mode in enumerate(MODES):
        analysis = run_analysis(df, "reason_text", mode, k)
        tasks = word_intrusion_tasks(analysis.result.keywords(10), n_words=5, seed=SEED + i)
        tasks["model"] = mode
        parts.append(tasks)
    tasks = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    tasks.insert(0, "id", [f"W{i:02d}" for i in range(1, len(tasks) + 1)])
    return tasks


def reason_tasks(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """60 ответов (по 6 на причину) и то, к какой причине их отнесла модель."""
    analysis = run_analysis(df, "reason_text", next(iter(MODES)), k)
    data = analysis.data[analysis.data["true_topic"].isin(REASONS)]
    mapping = majority_mapping(data["true_topic"], data[TOPIC_COL])
    data = data.assign(model_reason=data[TOPIC_COL].map(mapping))
    sample = (data.groupby("true_topic").sample(PER_REASON, random_state=SEED)
              .sample(frac=1, random_state=SEED).reset_index(drop=True))
    return pd.DataFrame({
        "id": [f"R{i:02d}" for i in range(1, len(sample) + 1)],
        "text": sample["clean_text"],
        "true_reason": sample["true_topic"].map(REASONS),
        "model_reason": sample["model_reason"].map(REASONS),
    })


def write_form(words: pd.DataFrame, reasons: pd.DataFrame, path: Path) -> None:
    """Excel с выпадающими списками: в каждой строке выбрать вариант из списка."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    intro = wb.active
    intro.title = "Инструкция"
    lines = [
        "Оценка тем людьми — заполняйте самостоятельно, не советуясь с остальными.",
        "",
        f"Лист «{SHEET_WORDS}»: в каждой строке шесть слов. Пять из них описывают одну тему, одно — лишнее.",
        "Выберите лишнее слово в колонке «Ваш ответ» (выпадающий список). Если не уверены — выберите наиболее вероятное.",
        "",
        f"Лист «{SHEET_REASONS}»: прочитайте ответ уволившегося сотрудника и выберите основную причину ухода.",
        "Если причин несколько — главную. Подсказок нет специально: важно ваше собственное мнение.",
        "",
        "Займёт 20–25 минут. Сохраните файл в data/human_eval/answers/ под своим именем, например stanislav.xlsx.",
        "Не открывайте data/human_eval/key.csv до того, как закончите: там правильные ответы.",
    ]
    for i, line in enumerate(lines, 1):
        intro.cell(i, 1, line)
    intro["A1"].font = Font(bold=True, size=13)
    intro.column_dimensions["A"].width = 120

    header_fill = PatternFill("solid", fgColor="CDE2FB")
    answer_fill = PatternFill("solid", fgColor="FFF6D5")

    def sheet(title: str, headers: list[str], widths: list[int]):
        ws = wb.create_sheet(title)
        ws.append(headers)
        for col, width in enumerate(widths, 1):
            ws.column_dimensions[chr(64 + col)].width = width
            ws.cell(1, col).font = Font(bold=True)
            ws.cell(1, col).fill = header_fill
        ws.freeze_panes = "A2"
        return ws

    ws = sheet(SHEET_WORDS, ["№", "Слова", "Ваш ответ"], [8, 70, 22])
    for r, row in enumerate(words.itertuples(), 2):
        ws.append([row.id, row.words, None])
        dv = DataValidation(type="list", formula1='"' + ",".join(w.strip() for w in row.words.split(",")) + '"',
                            allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"C{r}")
        ws.cell(r, 3).fill = answer_fill

    lists = wb.create_sheet("списки")
    for i, name in enumerate(REASONS.values(), 1):
        lists.cell(i, 1, name)
    lists.sheet_state = "hidden"

    ws = sheet(SHEET_REASONS, ["№", "Ответ сотрудника", "Основная причина"], [8, 100, 34])
    dv = DataValidation(type="list", formula1=f"='списки'!$A$1:$A${len(REASONS)}", allow_blank=True)
    ws.add_data_validation(dv)
    for r, row in enumerate(reasons.itertuples(), 2):
        ws.append([row.id, row.text, None])
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(r, 3).fill = answer_fill
        dv.add(f"C{r}")
    wb.save(path)


def make() -> None:
    df = load_table(LABELED)
    words, reasons = intrusion_tasks(df), reason_tasks(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS_DIR.mkdir(exist_ok=True)
    (ANSWERS_DIR / ".gitkeep").touch()
    write_form(words, reasons, FORM)
    key = pd.concat([
        words.rename(columns={"intruder": "answer"})[["id", "answer", "model", "topic", "words"]],
        reasons.rename(columns={"true_reason": "answer"})[["id", "answer", "model_reason", "text"]],
    ], ignore_index=True)
    key.to_csv(KEY, index=False, encoding="utf-8-sig")
    print(f"Бланк: {FORM.relative_to(ROOT)} ({len(words)} заданий «лишнее слово», {len(reasons)} ответов)")
    print(f"Ключ: {KEY.relative_to(ROOT)}")


# --- подсчёт ------------------------------------------------------------------------


def read_answers(path: Path) -> pd.Series:
    """Ответы одного участника: id задания → выбранный вариант."""
    sheets = pd.read_excel(path, sheet_name=[SHEET_WORDS, SHEET_REASONS], dtype=str)
    rows = pd.concat([s.iloc[:, [0, 2]].set_axis(["id", "answer"], axis=1) for s in sheets.values()])
    return rows.set_index("id")["answer"].fillna("").str.strip()


def score(answers_dir: Path = ANSWERS_DIR, key_path: Path = KEY, results_path: Path | None = RESULTS) -> pd.DataFrame:
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False).set_index("id")
    files = sorted(answers_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"В {answers_dir} нет заполненных бланков (*.xlsx)")
    people = {f.stem: read_answers(f).reindex(key.index).fillna("") for f in files}
    words, reasons = key[key.index.str.startswith("W")], key[key.index.str.startswith("R")]

    rows = []
    for name, ans in people.items():
        row = {"участник": name, "заполнено": f"{(ans != '').sum()} из {len(key)}"}
        for mode in MODES:
            ids = words.index[words["model"] == mode]
            row[f"лишнее слово: {mode}"] = round(float((ans[ids].str.lower() == words.loc[ids, "answer"].str.lower()).mean()), 2)
        row["причина: совпадение с эталоном"] = round(float((ans[reasons.index] == reasons["answer"]).mean()), 2)
        row["причина: каппа с моделью"] = round(annotator_agreement(ans[reasons.index], reasons["model_reason"]), 2)
        rows.append(row)
    model_row = {"участник": "модель (Qwen3)",
                 "причина: совпадение с эталоном": round(float((reasons["model_reason"] == reasons["answer"]).mean()), 2)}
    table = pd.DataFrame(rows + [model_row])

    pairs = [(a, b, annotator_agreement(people[a][reasons.index], people[b][reasons.index]))
             for a, b in itertools.combinations(people, 2)]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(table.to_string(index=False))
    if pairs:
        print("\nСогласие участников между собой (каппа Коэна, задание «причина»):")
        for a, b, kappa in pairs:
            print(f"  {a} — {b}: {kappa:.2f}")
        print(f"  в среднем: {np.mean([p[2] for p in pairs]):.2f}")
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(results_path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["make", "score"])
    args = parser.parse_args()
    random.seed(SEED)
    make() if args.command == "make" else score()


if __name__ == "__main__":
    main()
