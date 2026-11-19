import io

import pandas as pd
import pytest

from src.preprocessing import TextPreprocessor, guess_text_column, load_table, normalize_text, preprocess_dataframe


@pytest.fixture(scope="module")
def tp():
    return TextPreprocessor()


# --- лёгкая очистка ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  много   пробелов  ", "много пробелов"),
        ("Ужас!!! 😔", "Ужас!"),
        ("плохо :(", "плохо"),
        ("пишите hr@mail.ru", "пишите"),
        ("звоните +7 (999) 123-45-67", "звоните"),
        ("см. https://example.com", "см."),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_keeps_case_and_words():
    # Для эмбеддингов текст должен остаться «человеческим».
    assert normalize_text("Начальник ХАМИТ") == "Начальник ХАМИТ"


def test_normalize_fixes_all_caps():
    assert normalize_text("ВСЁ ПЛОХО, ЗАРПЛАТА МАЛЕНЬКАЯ") == "Всё плохо, зарплата маленькая"


# --- токены и леммы ------------------------------------------------------------


def test_lemmatization(tp):
    assert tp.tokens("платят мало, переработки не оплачиваются") == ["платить", "мало", "переработка", "оплачиваться"]


def test_stopwords_and_service_words_removed(tp):
    tokens = tp.tokens("Я ушла из компании, потому что очень устала от этой работы")
    assert "я" not in tokens and "из" not in tokens and "очень" not in tokens
    assert "компания" not in tokens and "работа" not in tokens  # доменные стоп-слова
    assert "устать" in tokens


def test_case_and_yo_normalized(tp):
    assert tp.tokens("УДАЛЁНКУ отменили") == tp.tokens("удаленку отменили")


def test_yo_helps_pronoun_parsing(tp):
    # «в нём» — местоимение, а не прилагательное «немой».
    assert "немой" not in tp.tokens("не вижу в нём ценности")


def test_slang_and_abbreviations(tp):
    assert tp.tokens("зп маленькая")[0] == "зарплата"
    assert tp.tokens("з/п задерживают")[0] == "зарплата"
    assert tp.tokens("уровень заработной платы") == ["уровень", "зарплата"]


def test_hyphenated_words_kept(tp):
    assert "колл-центр" in tp.tokens("работа в колл-центре")


def test_join_negations():
    tp = TextPreprocessor(join_negations=True)
    assert tp.tokens("мне не платят") == ["не_платить"]


def test_domain_stopwords_can_be_disabled():
    assert "работа" in TextPreprocessor(use_domain_stopwords=False).tokens("тяжёлая работа")


def test_extra_stopwords():
    assert "склад" not in TextPreprocessor(extra_stopwords=["склад"]).tokens("холодно на складе")


# --- бессмысленные ответы ------------------------------------------------------


@pytest.mark.parametrize("text", ["нет", "-", "...", "???", "N/A", "Без комментариев", "ВСЁ ХОРОШО", "", None,
                                  "Просто решил уйти"])
def test_meaningless_answers(tp, text):
    assert not tp.is_meaningful(text)


@pytest.mark.parametrize("text", ["мало платят", "Переезд", "токсичный коллектив", "нет роста"])
def test_meaningful_answers(tp, text):
    assert tp.is_meaningful(text)


# --- таблица целиком -----------------------------------------------------------


def test_preprocess_dataframe_drops_noise_and_duplicates():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "dept": ["IT", "IT", "IT", "HR"],
        "text": ["Мало платят!!!", "Мало платят!!!", "нет", "Токсичный коллектив"],
    })
    out, report = preprocess_dataframe(df, "text")
    assert report.n_duplicates == 1  # строки 1 и 2 совпадают во всём, кроме id
    assert report.n_empty == 1
    assert list(out["id"]) == ["1", "4"]
    assert out.loc[0, "clean_text"] == "Мало платят!"
    assert out.loc[0, "tokens"] == ["мало", "платить"]


def test_same_text_from_different_people_is_not_duplicate():
    df = pd.DataFrame({"id": ["1", "2"], "dept": ["IT", "HR"], "text": ["Мало платят", "Мало платят"]})
    _, report = preprocess_dataframe(df, "text")
    assert report.n_duplicates == 0


# --- загрузка файлов -----------------------------------------------------------


def _file(data: bytes, name: str) -> io.BytesIO:
    f = io.BytesIO(data)
    f.name = name
    return f


def test_load_csv_semicolon_cp1251():
    raw = "id;ответ\n1;мало платят\n2;n/a\n".encode("cp1251")
    df = load_table(_file(raw, "export.csv"))
    assert list(df.columns) == ["id", "ответ"]
    assert df.loc[1, "ответ"] == "n/a"  # не превратился в NaN


def test_load_excel(tmp_path):
    path = tmp_path / "a.xlsx"
    pd.DataFrame({"id": [1], "text": ["мало платят"]}).to_excel(path, index=False)
    assert load_table(path).loc[0, "text"] == "мало платят"


def test_guess_text_column():
    df = pd.DataFrame({"id": ["1", "2"], "dept": ["IT", "HR"], "answer": ["длинный ответ про зарплату", "ещё ответ"]})
    assert guess_text_column(df) == "answer"
