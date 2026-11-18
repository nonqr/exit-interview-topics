"""Загрузка таблицы с ответами из CSV или Excel.

HR-выгрузки бывают в разных форматах: CSV с запятой или точкой с запятой,
в кодировке UTF-8 или Windows-1251, либо Excel. Функция load_table
разбирается с этим сама, чтобы пользователю не пришлось ничего настраивать.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

ENCODINGS = ("utf-8-sig", "cp1251")


def load_table(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Читает CSV/XLSX. source — путь или файловый объект (например, из st.file_uploader).

    Все значения читаются как строки, а «n/a», «-» и т.п. НЕ превращаются в пустые
    значения: решение о том, что ответ пустой, принимает предобработка.
    """
    name = str(filename or getattr(source, "name", source)).lower()
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)

    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось определить кодировку файла (ожидается UTF-8 или Windows-1251)")

    try:
        sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def guess_text_column(df: pd.DataFrame) -> str:
    """Колонка со свободным текстом — та, где строки в среднем самые длинные."""
    if df.empty or len(df.columns) == 0:
        raise ValueError("Таблица пустая")
    return max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
