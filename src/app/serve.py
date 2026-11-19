"""Запуск сервиса с моделями, заранее загруженными в память (Kaggle, Colab, показ на защите).

    python -m src.app.serve                      # загрузить обе модели и запустить сервис
    python -m src.app.serve --preload            # без предзагрузки (как обычный streamlit run)

Streamlit выполняет страницу в том же процессе, поэтому модель, загруженная здесь,
остаётся в памяти: первый же загруженный файл считается сразу, без ожидания
загрузки 8 ГБ весов.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.models.embedding import DEFAULT_MODEL, LIGHT_MODEL, Encoder, pick_device

APP = Path(__file__).with_name("streamlit_app.py")


def preload(model_name: str) -> None:
    t0 = time.time()
    encoder = Encoder(model_name, use_cache=False)
    model = encoder._model()
    encoder.encode(["Пробный ответ: ушёл из-за низкой зарплаты."])  # прогрев: первый вызов самый медленный
    print(f"Модель {model_name} загружена на {model.device} за {time.time() - t0:.0f} с", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preload", nargs="*", default=[DEFAULT_MODEL, LIGHT_MODEL],
                        help="какие модели загрузить заранее (по умолчанию точная и быстрая)")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    print(f"Устройство для эмбеддингов: {pick_device()}", flush=True)
    for model_name in args.preload:
        preload(model_name)

    from streamlit.web import cli

    sys.argv = [
        "streamlit", "run", str(APP), f"--server.port={args.port}", "--server.headless=true",
        # Сервис открывается через туннель с другого адреса: проверки происхождения запроса мешают загрузке файлов.
        "--server.enableCORS=false", "--server.enableXsrfProtection=false",
        "--server.fileWatcherType=none", "--browser.gatherUsageStats=false",
    ]
    print(f"Запускаю веб-сервер на порту {args.port}", flush=True)
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
