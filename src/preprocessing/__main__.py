"""Запуск предобработки из командной строки (демонстрация для показа 2).

    python -m src.preprocessing data/raw/exit_interviews.csv
    python -m src.preprocessing data/raw/exit_interviews.xlsx --examples 10 --join-negations

Печатает отчёт и примеры «грязный текст → чистые леммы», сохраняет результат
в data/processed/<имя файла>_processed.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.io import guess_text_column, load_table
from src.preprocessing.pipeline import preprocess_dataframe
from src.preprocessing.text import TextPreprocessor

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Предобработка ответов экзит-интервью")
    parser.add_argument("path", type=Path, help="CSV или XLSX с ответами")
    parser.add_argument("--column", help="колонка с текстом (по умолчанию определяется автоматически)")
    parser.add_argument("--examples", type=int, default=8, help="сколько примеров показать")
    parser.add_argument("--join-negations", action="store_true", help="склеивать «не» со следующим словом")
    parser.add_argument("--no-domain-stopwords", action="store_true", help="не удалять доменные стоп-слова")
    args = parser.parse_args()

    df = load_table(args.path)
    column = args.column or guess_text_column(df)
    tp = TextPreprocessor(
        use_domain_stopwords=not args.no_domain_stopwords,
        join_negations=args.join_negations,
    )
    processed, report = preprocess_dataframe(df, column, tp)

    print(f"Колонка с текстом: {column}\n")
    print(report.as_text())

    print("\nПримеры:")
    # Сначала показываем самые «грязные» строки — так нагляднее.
    dirty = processed[processed[column] != processed["clean_text"]]
    sample = (dirty if len(dirty) >= args.examples else processed).head(args.examples)
    for _, row in sample.iterrows():
        print(f"\n  было:  {row[column]!r}")
        print(f"  стало: {row['tokens']}")

    dropped = df.loc[~df[column].map(tp.is_meaningful), column].head(args.examples).tolist()
    print(f"\nОтброшены как бессмысленные (первые {len(dropped)}): {dropped}")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.path.stem}_processed.csv"
    processed.drop(columns=["tokens"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nРезультат сохранён: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
