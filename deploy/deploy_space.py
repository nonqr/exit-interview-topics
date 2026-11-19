"""Публикация веб-сервиса на Hugging Face Spaces.

    HF_TOKEN=hf_... python deploy/deploy_space.py <пользователь>/<имя-space>

Собирает во временной папке только то, что нужно сервису (код, демо-датасет,
Dockerfile, карточку Space), и загружает на Space. Файлы, которых больше нет
в проекте, удаляются со Space. Space создаётся, если его ещё нет.
Hugging Face сам соберёт Docker-образ и запустит сервис.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "Dockerfile",
    ".dockerignore",
    "requirements-app.txt",
    ".streamlit/config.toml",
    "src",
    "data/raw/exit_interviews.xlsx",
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def stage(target: Path) -> None:
    for item in INCLUDE:
        src, dst = ROOT / item, target / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
    # У Space своя карточка: в ней настройки (sdk: docker, порт) и описание.
    shutil.copy2(ROOT / "deploy" / "space_README.md", target / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", help="идентификатор Space, например nonqr/exit-interview-topics")
    parser.add_argument("--message", default="Обновление сервиса", help="описание изменения")
    parser.add_argument("--dry-run", action="store_true", help="только собрать папку и показать файлы")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        stage(folder)
        files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        print(f"Файлов к загрузке: {len(files)}")
        if args.dry_run:
            print("\n".join(files))
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Задайте токен Hugging Face в переменной окружения HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=args.space, repo_type="space",
                          commit_message=args.message, delete_patterns="*")
        print(f"Готово: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
