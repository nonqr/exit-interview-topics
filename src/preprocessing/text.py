"""Предобработка текста ответов экзит-интервью.

Два уровня обработки:

1. normalize_text() — «лёгкая» очистка без изменения смысла: убирает эмодзи,
   ссылки, лишние пробелы и повторы знаков. Такой текст подаётся в модель
   эмбеддингов (e5): ей важны полные формы слов и порядок, лемматизация вредит.

2. TextPreprocessor — полная обработка для LDA / NMF и для ключевых слов тем:
   нижний регистр → токенизация → лемматизация (pymorphy3) → фильтр по части
   речи → стоп-слова. На выходе список чистых лемм.

Пример:
    >>> tp = TextPreprocessor()
    >>> tp.tokens("Зп маленькая, а обязанностей добавили в два раза больше!!! 😔")
    ['зарплата', 'маленький', 'обязанность', 'добавить']
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pymorphy3

RESOURCES = Path(__file__).parent / "resources"

# --- регулярные выражения для лёгкой очистки ---------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")
# Эмодзи и прочие пиктограммы: всё, что не буква, цифра, пробел или обычная пунктуация.
EMOJI_RE = re.compile(r"[^\w\s.,;:!?()«»\"'%/№+\-–—]", re.UNICODE)
EMOTICON_RE = re.compile(r"(?:^|\s)[:;=]-?[()DPp]+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")
SPACES_RE = re.compile(r"\s+")

# Устойчивые сочетания, которые надо склеить до токенизации.
PHRASE_REPLACEMENTS = [
    (re.compile(r"\bз\s*/\s*п\b", re.I), "зарплата"),
    (re.compile(r"\bзаработн\w*\s+плат\w*", re.I), "зарплата"),
]

# Токен — последовательность букв, допускается дефис внутри («колл-центр»).
TOKEN_RE = re.compile(r"[a-zа-яё]+(?:-[a-zа-яё]+)*")

# Части речи (теги OpenCorpora), которые несут смысл. Остальные — служебные:
# местоимения (NPRO), предлоги (PREP), союзы (CONJ), частицы (PRCL), междометия (INTJ),
# компаративы (COMP: «больше», «лучше»).
CONTENT_POS = {
    "NOUN",  # существительное: зарплата, руководитель
    "ADJF", "ADJS",  # прилагательное: токсичный, низкий
    "VERB", "INFN",  # глагол: платить, выгореть
    "PRTF", "PRTS", "GRND",  # причастие, деепричастие
    "ADVB",  # наречие: скучно, далеко
    "PRED",  # предикатив: нельзя, тяжело
    "NUMR",  # pymorphy считает «мало», «много» числительными
    None,  # неизвестные слова: сокращения (зп, дмс), латиница (kpi)
}

# Типичные «пустые» ответы в анкетах. Сравниваются с нормализованным текстом целиком.
# Список составлен заранее по смыслу, а не скопирован из нашего датасета, чтобы
# оценка фильтра на датасете была честной. Ответы, где после удаления стоп-слов
# не остаётся ни одного слова («просто решил уйти»), отсекаются и без этого списка.
NON_ANSWERS = {
    "нет", "да", "ок", "ok", "n a", "na", "пусто", "не знаю", "без комментариев",
    "без причины", "не хочу отвечать", "нечего добавить", "затрудняюсь ответить",
    "по собственному желанию", "все хорошо", "все нормально", "все было нормально",
    "все устраивало", "не важно", "неважно", "спасибо", "спасибо за все", "удачи",
}


def _read_list(name: str) -> list[str]:
    lines = (RESOURCES / name).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def load_stopwords(include_domain: bool = True) -> set[str]:
    words = set(_read_list("stopwords_ru.txt"))
    if include_domain:
        words |= set(_read_list("stopwords_domain.txt"))
    return words


def load_lemma_overrides() -> dict[str, str]:
    pairs = (ln.split("\t") for ln in _read_list("lemma_overrides.txt"))
    return {src: dst for src, dst in pairs}


def normalize_text(text: object) -> str:
    """Лёгкая очистка: регистр и слова не меняются. Пустые значения → ''."""
    if text is None or (isinstance(text, float) and text != text):  # None или NaN
        return ""
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)  # телефоны — персональные данные
    text = EMOTICON_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = SPACES_RE.sub(" ", text).strip()
    if _is_shouting(text):
        # «ВСЁ ПЛОХО» → «Всё плохо»: нейросетевые модели воспринимают капслок как
        # отдельный «стиль» и группируют такие ответы вместе независимо от смысла.
        text = text[:1].upper() + text[1:].lower()
    return text


def _is_shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 3 and sum(c.isupper() for c in letters) / len(letters) > 0.7


def _canonical(text: str) -> str:
    """Нижний регистр, «ё» → «е», только буквы и пробелы — для сравнения ответов."""
    return " ".join(TOKEN_RE.findall(text.lower().replace("ё", "е")))


@dataclass
class TextPreprocessor:
    """Полная предобработка: токенизация, лемматизация, фильтрация.

    Параметры:
        use_domain_stopwords — убирать слова, общие для всех причин («работа», «уйти»);
        pos_filter          — оставлять только значимые части речи (см. CONTENT_POS);
        join_negations      — склеивать «не» со следующим словом: «не платят» → «не_платить».
                              Сохраняет отрицание, но делает словарь более разреженным;
        min_token_len       — отбрасывать слишком короткие токены;
        extra_stopwords     — дополнительные стоп-слова (например, из интерфейса).
    """

    use_domain_stopwords: bool = True
    pos_filter: bool = True
    join_negations: bool = False
    min_token_len: int = 2
    extra_stopwords: Iterable[str] = ()
    stopwords: set[str] = field(init=False)
    overrides: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.stopwords = load_stopwords(self.use_domain_stopwords) | {w.lower() for w in self.extra_stopwords}
        self.overrides = load_lemma_overrides()
        self._morph = _get_morph()

    # --- отдельные шаги, их удобно показывать на демонстрации -----------------

    def tokenize(self, text: object) -> list[str]:
        """Нижний регистр, склейка устойчивых сочетаний, выделение слов.

        «ё» здесь сохраняется: она помогает pymorphy выбрать правильный разбор
        («в нём» → «он», а не «немой»). На «е» она заменяется уже в лемме.
        """
        text = normalize_text(text).lower()
        for pattern, repl in PHRASE_REPLACEMENTS:
            text = pattern.sub(repl, text)
        return TOKEN_RE.findall(text)

    def lemmatize(self, token: str) -> tuple[str, str | None]:
        """Начальная форма слова и его часть речи (кэшируется)."""
        lemma, pos = _parse(token)
        lemma = lemma.replace("ё", "е")
        return self.overrides.get(lemma, lemma), pos

    def tokens(self, text: object) -> list[str]:
        """Весь конвейер: текст → список чистых лемм."""
        result: list[str] = []
        negate = False
        for token in self.tokenize(text):
            lemma, pos = self.lemmatize(token)
            if self.join_negations and lemma == "не":
                negate = True
                continue
            if self.pos_filter and pos not in CONTENT_POS:
                negate = False
                continue
            if lemma in self.stopwords or len(lemma) < self.min_token_len:
                negate = False
                continue
            result.append(f"не_{lemma}" if negate else lemma)
            negate = False
        return result

    def lemmatized_text(self, text: object) -> str:
        """Леммы через пробел — удобный формат для CountVectorizer / TfidfVectorizer."""
        return " ".join(self.tokens(text))

    def is_meaningful(self, text: object) -> bool:
        """False для пустых и бессмысленных ответов («нет», «-», «всё ок»)."""
        normalized = normalize_text(text)
        if not TOKEN_RE.search(normalized.lower()):
            return False  # нет ни одной буквы: «-», «...», «???»
        if _canonical(normalized) in NON_ANSWERS:
            return False
        return len(self.tokens(normalized)) > 0


@lru_cache(maxsize=1)
def _get_morph() -> pymorphy3.MorphAnalyzer:
    # Создание анализатора занимает ~0.5 с и много памяти — делаем один на процесс.
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _parse(token: str) -> tuple[str, str | None]:
    # В ответах много повторяющихся слов, кэш ускоряет обработку в разы.
    p = _get_morph().parse(token)[0]
    return p.normal_form, p.tag.POS
