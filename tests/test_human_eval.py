"""Бланк оценки людьми читается, и подсчёт даёт ожидаемые числа."""

import random
import shutil

import pandas as pd
from openpyxl import load_workbook

from src.evaluation.human_eval import FORM, KEY, REASONS, SHEET_REASONS, SHEET_WORDS, score


def fill(path, answers):
    wb = load_workbook(path)
    for sheet in (SHEET_WORDS, SHEET_REASONS):
        ws = wb[sheet]
        for row in range(2, ws.max_row + 1):
            ws.cell(row, 3, answers(ws.cell(row, 1).value, ws.cell(row, 2).value))
    wb.save(path)


def test_form_has_dropdowns():
    wb = load_workbook(FORM)
    assert wb[SHEET_WORDS].data_validations.dataValidation
    assert wb[SHEET_REASONS].data_validations.dataValidation
    assert wb["списки"].sheet_state == "hidden"


def test_score_perfect_and_random(tmp_path):
    key = pd.read_csv(KEY, keep_default_na=False).set_index("id")["answer"]
    shutil.copy(FORM, tmp_path / "perfect.xlsx")
    shutil.copy(FORM, tmp_path / "random.xlsx")
    fill(tmp_path / "perfect.xlsx", lambda i, _: key[i])
    rng = random.Random(0)
    fill(tmp_path / "random.xlsx",
         lambda i, words: rng.choice(words.split(", ")) if i.startswith("W") else rng.choice(list(REASONS.values())))

    table = score(tmp_path, KEY, results_path=None).set_index("участник")
    perfect, rnd = table.loc["perfect"], table.loc["random"]
    assert perfect["заполнено"] == f"{len(key)} из {len(key)}"
    assert all(perfect[c] == 1.0 for c in table.columns if c.startswith("лишнее слово"))
    assert perfect["причина: совпадение с эталоном"] == 1.0
    assert rnd["причина: совпадение с эталоном"] < 0.4
