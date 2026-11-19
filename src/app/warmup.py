"""Подготовка сервиса к запуску: скачать модели и посчитать эмбеддинги демо-датасета.

    python -m src.app.warmup

Запускается при сборке Docker-образа. После него первый анализ демо-датасета
проходит за секунды, а не за минуты, и заодно проверяется, что весь конвейер
работает: если что-то сломано, сборка образа упадёт, а не сервис у пользователя.
"""

from __future__ import annotations

import time

from src.app.analysis import DEMO_FILE, MODES, run_analysis
from src.app.serve import preload
from src.preprocessing import guess_text_column, load_table


def main() -> None:
    # Эмбеддинги демо уже лежат в data/embeddings/, поэтому анализ демо модели не загрузит:
    # скачиваем модели явно, чтобы сервис не качал их при первом новом файле.
    for model_name in filter(None, MODES.values()):
        preload(model_name)
    df = load_table(DEMO_FILE)
    text_col = guess_text_column(df)
    for mode, model_name in MODES.items():
        t0 = time.time()
        analysis = run_analysis(df, text_col, mode, n_topics=10, progress=print)
        print(f"{mode}: {analysis.result.n_topics} тем по {len(analysis.data)} ответам, "
              f"{time.time() - t0:.0f} с ({model_name or 'без нейросети'})", flush=True)


if __name__ == "__main__":
    main()
