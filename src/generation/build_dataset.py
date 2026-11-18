"""Сборка синтетического датасета экзит-интервью.

Исходные тексты лежат в data/synthetic_source/<topic>.txt (одна запись на строку,
необязательный префикс [код] — вторая причина в том же ответе). Скрипт:

1. читает тексты и метки;
2. добавляет правдоподобные метаданные (отдел, стаж, должность, дата и т.д.)
   со смещениями по темам — чтобы в дашборде были реальные закономерности;
3. «загрязняет» часть текстов (регистр, лишние пробелы, эмодзи, дубли), чтобы
   было что чистить на этапе предобработки;
4. сохраняет два варианта:
   - data/raw/exit_interviews.csv/.xlsx — то, что HR загружает в сервис (без меток);
   - data/raw/exit_interviews_labeled.csv — с эталонными метками для оценки качества.

Запуск:  python -m src.generation.build_dataset
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.generation.taxonomy import (
    BASE_DEPARTMENT_WEIGHTS,
    BASE_TENURE_WEIGHTS,
    NOISE_CODE,
    NOISE_NAME,
    TOPIC_BY_CODE,
    TOPICS,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "synthetic_source"
RAW_DIR = ROOT / "data" / "raw"
SEED = 42

SECONDARY_RE = re.compile(r"^\[(\w+)\]\s*(.+)$")

POSITIONS = {
    "IT": ["Junior-разработчик", "Разработчик", "Senior-разработчик", "Тестировщик", "Аналитик", "Тимлид"],
    "Продажи": ["Менеджер по продажам", "Старший менеджер по продажам", "Аккаунт-менеджер", "Руководитель группы продаж"],
    "Колл-центр": ["Оператор", "Старший оператор", "Супервайзер"],
    "Логистика и склад": ["Кладовщик", "Комплектовщик", "Водитель-экспедитор", "Логист", "Начальник смены"],
    "Финансы и бухгалтерия": ["Бухгалтер", "Главный специалист", "Финансовый аналитик", "Экономист"],
    "Маркетинг": ["Маркетолог", "SMM-специалист", "Дизайнер", "Руководитель направления"],
    "HR": ["Рекрутер", "HR-бизнес-партнёр", "Специалист по кадрам"],
    "Производство": ["Оператор линии", "Слесарь", "Мастер участка", "Инженер-технолог"],
}

WORK_FORMATS = {
    "IT": {"Удалённо": 0.35, "Гибрид": 0.4, "Офис": 0.25},
    "Маркетинг": {"Удалённо": 0.2, "Гибрид": 0.45, "Офис": 0.35},
    "HR": {"Удалённо": 0.1, "Гибрид": 0.4, "Офис": 0.5},
    "Финансы и бухгалтерия": {"Удалённо": 0.1, "Гибрид": 0.35, "Офис": 0.55},
    "Продажи": {"Удалённо": 0.1, "Гибрид": 0.3, "Офис": 0.6},
}
DEFAULT_WORK_FORMAT = {"Офис": 1.0}  # колл-центр, склад, производство

TENURE_MONTHS = {"<6 мес": (1, 5), "6-12 мес": (6, 12), "1-3 года": (13, 36), "3-5 лет": (37, 60), ">5 лет": (61, 180)}

# Правила согласования метаданных с текстом: если в ответе упомянут склад,
# смены или релизы, отдел должен быть правдоподобным. Первое совпадение побеждает.
DEPARTMENT_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"склад|погрузк|пересорт|спецодежд|экспедит", re.I), ["Логистика и склад"]),
    (re.compile(r"цех|конвейер|производств|вредност|хими|беруши|сдельн|выработк", re.I),
     ["Производство", "Логистика и склад"]),
    (re.compile(r"колл-центр|гарнитур|звонк|скрипт|обращени", re.I), ["Колл-центр"]),
    (re.compile(r"\bсмен(а|ы|у|е|ах|ами)?\b|сменн|\bвахт|ночн", re.I), ["Колл-центр", "Логистика и склад", "Производство"]),
    (re.compile(r"релиз|бэклог|инцидент|тимлид|технологи", re.I), ["IT"]),
    (re.compile(r"маркетолог|рассылк|творческ", re.I), ["Маркетинг"]),
    (re.compile(r"кадровые документы", re.I), ["HR"]),
    (re.compile(r"клиент|план повыша|процент снижают", re.I), ["Продажи", "Колл-центр"]),
    (re.compile(r"удалён|удален|из дома|гибрид", re.I),
     ["IT", "Маркетинг", "HR", "Финансы и бухгалтерия", "Продажи"]),
]
REMOTE_LOST_RE = re.compile(r"удалён|удален|из дома|гибрид|в офис", re.I)
MANAGER_POSITION_RE = re.compile(r"Руковод|Тимлид|Начальник|Супервайзер|Мастер")

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def skewed(base: dict[str, float], skew: dict[str, float]) -> dict[str, float]:
    """Умножает базовые веса на коэффициенты темы."""
    return {k: v * skew.get(k, 1.0) for k, v in base.items()}


def read_source() -> list[dict]:
    records = []
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        topic = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            secondary = None
            m = SECONDARY_RE.match(line)
            if m:
                secondary, line = m.group(1), m.group(2)
                assert secondary in TOPIC_BY_CODE, f"{path.name}: неизвестная тема [{secondary}]"
            records.append({"reason_text": line, "true_topic": topic, "secondary_topic": secondary})
    return records


def random_date(rng: random.Random, topic: str) -> date:
    # Закономерность для дашборда: после отмены удалёнки (март 2025) растёт доля
    # увольнений из-за графика и формата работы.
    if topic == "schedule" and rng.random() < 0.7:
        start = date(2025, 3, 1)
    else:
        start = START_DATE
    span = (END_DATE - start).days
    return start + timedelta(days=rng.randint(0, span))


def satisfaction(rng: random.Random, topic: str) -> int:
    """Общая оценка работы в компании 1–5: личные причины — выше, конфликты — ниже."""
    mean = {"personal": 4.0, NOISE_CODE: 3.5, "schedule": 3.2, "career": 3.0, "content": 2.9,
            "salary": 2.8, "conditions": 2.5, "processes": 2.5, "workload": 2.3,
            "team": 2.2, "management": 1.9}[topic]
    return int(min(5, max(1, round(rng.gauss(mean, 0.8)))))


def add_metadata(rng: random.Random, rec: dict) -> dict:
    topic = TOPIC_BY_CODE.get(rec["true_topic"])
    dept_w = skewed(BASE_DEPARTMENT_WEIGHTS, topic.department_weights if topic else {})
    ten_w = skewed(BASE_TENURE_WEIGHTS, topic.tenure_weights if topic else {})
    text = rec["reason_text"]
    for pattern, allowed in DEPARTMENT_HINTS:
        if pattern.search(text):
            dept_w = {d: w for d, w in dept_w.items() if d in allowed}
            break
    dept = weighted_choice(rng, dept_w)

    tenure = weighted_choice(rng, ten_w)
    age = max(19, min(64, int(rng.gauss(33, 8))))
    if re.search(r"\bпенси", text, re.I):
        tenure, age = ">5 лет", rng.randint(58, 64)
    elif re.search(r"призыв|магистратур|аспирантур|закончить вуз|учёб|учеб", text, re.I):
        age = rng.randint(19, 27)
        if tenure in (">5 лет", "3-5 лет"):
            tenure = "1-3 года"
    lo, hi = TENURE_MONTHS[tenure]

    positions = POSITIONS[dept]
    if rec["true_topic"] == "career":  # жалуется на отсутствие роста — значит, не руководитель
        positions = [p for p in positions if not MANAGER_POSITION_RE.search(p)] or positions

    work_format = weighted_choice(rng, WORK_FORMATS.get(dept, DEFAULT_WORK_FORMAT))
    if REMOTE_LOST_RE.search(text):  # удалёнку отменили или не дают — сейчас человек в офисе
        work_format = "Офис"

    sat = satisfaction(rng, rec["true_topic"])
    rec.update(
        exit_date=random_date(rng, rec["true_topic"]).isoformat(),
        department=dept,
        position=rng.choice(positions),
        tenure_group=tenure,
        tenure_months=rng.randint(lo, hi),
        age=age,
        work_format=work_format,
        satisfaction=sat,
        would_recommend="Да" if rng.random() < (sat - 1) / 4 * 0.9 else "Нет",
    )
    return rec


def make_dirty(rng: random.Random, text: str) -> str:
    """Типичный «мусор» из реальных анкет — его должна убрать предобработка."""
    r = rng.random()
    if r < 0.05:
        return text.upper()
    if r < 0.12:
        return "  " + text.replace(" ", "  ", 2) + "  "
    if r < 0.17:
        return text.rstrip(".") + "!!!"
    if r < 0.21:
        return text + rng.choice([" 😔", " 🙁", " 👎", " :("])
    if r < 0.25:
        return text.replace("ё", "е")
    if r < 0.28:
        return text.lower().replace(",", "")
    return text


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    records = read_source()

    rows = []
    for rec in records:
        rec = add_metadata(rng, rec)
        rec["reason_text"] = make_dirty(rng, rec["reason_text"])
        rows.append(rec)

    # Несколько точных дублей строк: в реальных выгрузках так выглядит двойная отправка формы.
    rows += [dict(r) for r in rng.sample([r for r in rows if r["true_topic"] != NOISE_CODE], 6)]

    df = pd.DataFrame(rows).sort_values("exit_date").reset_index(drop=True)
    df.insert(0, "id", [f"EX-{i:04d}" for i in range(1, len(df) + 1)])
    names = {t.code: t.name for t in TOPICS} | {NOISE_CODE: NOISE_NAME}
    df["true_topic_name"] = df["true_topic"].map(names)
    df["secondary_topic_name"] = df["secondary_topic"].map(names)
    return df


def main() -> None:
    df = build()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    label_cols = ["true_topic", "true_topic_name", "secondary_topic", "secondary_topic_name"]
    public = df.drop(columns=label_cols)

    # utf-8-sig — чтобы Excel корректно открыл кириллицу в CSV.
    public.to_csv(RAW_DIR / "exit_interviews.csv", index=False, encoding="utf-8-sig")
    public.to_excel(RAW_DIR / "exit_interviews.xlsx", index=False)
    df.to_csv(RAW_DIR / "exit_interviews_labeled.csv", index=False, encoding="utf-8-sig")

    print(f"Записей: {len(df)}")
    print(df["true_topic_name"].value_counts().to_string())
    print(f"С двумя причинами: {df['secondary_topic'].notna().sum()}")


if __name__ == "__main__":
    main()
