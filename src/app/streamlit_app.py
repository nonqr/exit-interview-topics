"""Веб-интерфейс: анализ причин увольнения по ответам экзит-интервью.

Запуск из корня репозитория:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # streamlit запускает файл как скрипт — добавляем корень проекта
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.app import charts  # noqa: E402
from src.app.analysis import (  # noqa: E402
    CONF_COL, DEFAULT_MODE, DEMO_FILE, LOW_CONFIDENCE, MODES, NAME_COL, TOPIC_COL, Analysis, apply_names, date_columns,
    lift_by_segment, notable, run_analysis, segment_columns, share_by_segment, timeline, to_excel,
)
from src.preprocessing import guess_text_column, load_table  # noqa: E402

st.set_page_config(page_title="Причины увольнений", page_icon="📋", layout="wide")


# --- загрузка и анализ (с кэшем) --------------------------------------------------


@st.cache_data(show_spinner=False)
def read_file(data: bytes, name: str) -> pd.DataFrame:
    import io

    f = io.BytesIO(data)
    f.name = name
    return load_table(f)


@st.cache_data(show_spinner=False, max_entries=10)
def analyse(file_hash: str, _df: pd.DataFrame, text_col: str, mode: str, n_topics: int | None,
            stopwords: tuple[str, ...]) -> Analysis:
    # file_hash — ключ кэша: тот же файл с теми же настройками не пересчитывается.
    return run_analysis(_df, text_col, mode, n_topics, stopwords)


# --- боковая панель: данные и настройки --------------------------------------------

st.sidebar.header("1. Данные")
uploaded = st.sidebar.file_uploader("Файл с ответами (CSV или Excel)", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.toggle("Использовать демо-датасет", value=uploaded is None,
                             help="490 синтетических анкет экзит-интервью")

if uploaded is not None and not use_demo:
    raw, file_name = uploaded.getvalue(), uploaded.name
elif use_demo:
    raw, file_name = DEMO_FILE.read_bytes(), DEMO_FILE.name
else:
    raw = None

st.title("Причины увольнений по экзит-интервью")
st.caption("Загрузите выгрузку анкет — сервис очистит ответы, найдёт основные причины ухода и покажет, "
           "в каких отделах и когда они встречаются чаще.")

if raw is None:
    st.info("Загрузите файл в панели слева или включите демо-датасет.")
    st.stop()

try:
    df = read_file(raw, file_name)
except Exception as e:  # noqa: BLE001 — показываем пользователю понятную ошибку
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

columns = list(df.columns)
guessed = guess_text_column(df)
text_col = st.sidebar.selectbox("Колонка со свободным ответом", columns, index=columns.index(guessed))

st.sidebar.header("2. Настройки")
mode = st.sidebar.radio("Режим анализа", list(MODES), index=list(MODES).index(DEFAULT_MODE),
                        help="Точный режим лучше всего находит причины, но первый запуск скачивает модель "
                             "(~8 ГБ). Быстрый — модель ~2 ГБ, качество ниже. Без нейросети — мгновенно, "
                             "но темы заметно хуже.")
auto_k = st.sidebar.toggle("Подобрать число тем автоматически", value=False,
                           help="По силуэтному коэффициенту. Это подсказка: результат можно поправить вручную.")
n_topics = None if auto_k else st.sidebar.slider("Число тем", 3, 20, 10)
stop_text = st.sidebar.text_input("Дополнительные стоп-слова", placeholder="например: банк, филиал",
                                  help="Слова, которые не должны попадать в ключевые слова тем")
stopwords = tuple(sorted({w.strip().lower() for w in stop_text.split(",") if w.strip()}))

if st.sidebar.button("Анализировать", type="primary", use_container_width=True):
    file_hash = hashlib.sha1(raw).hexdigest()
    try:
        with st.spinner("Идёт анализ… Первый запуск скачивает модель — это может занять несколько минут."):
            st.session_state.analysis = analyse(file_hash, df, text_col, mode, n_topics, stopwords)
        st.session_state.pop("names_editor", None)
    except ValueError as e:
        st.error(str(e))

analysis: Analysis | None = st.session_state.get("analysis")
if analysis is None:
    st.subheader("Предпросмотр файла")
    st.write(f"Строк: **{len(df)}**, колонка с ответами: **{text_col}**")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.info("Нажмите «Анализировать» в панели слева.")
    st.stop()

data = analysis.data
tab_overview, tab_topics, tab_segments, tab_answers = st.tabs(["Обзор", "Темы", "Срезы", "Ответы"])

# --- Обзор -----------------------------------------------------------------------

with tab_overview:
    table = analysis.topic_table()
    top = table.iloc[0]
    low = int((data[CONF_COL] < LOW_CONFIDENCE).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Проанализировано ответов", f"{len(data)}", help="Без пустых ответов и повторных анкет")
    c2.metric("Найдено тем", f"{table.shape[0]}")
    c3.metric("Самая частая причина", f"{top['Доля']:.0%}", help=top["Тема"])
    c4.metric("Нужно перепроверить", f"{low}", help=f"Ответы, где уверенность модели ниже {LOW_CONFIDENCE:.0%}")
    st.markdown(f"**Самая частая причина:** {top['Тема']} — {top['Ответов']} ответов.")

    st.subheader("Распределение причин увольнения")
    st.plotly_chart(charts.topic_shares(table), use_container_width=True)

    r = analysis.report
    with st.expander("Как очищены данные"):
        st.markdown(
            f"- Строк в файле: **{r.n_input}**\n"
            f"- Пустые и бессмысленные ответы («нет», «-», «без комментариев»): **{r.n_empty}**\n"
            f"- Повторно отправленные анкеты: **{r.n_duplicates}**\n"
            f"- Проанализировано: **{r.n_output}**\n"
            f"- Режим: **{analysis.mode}**"
        )
    if analysis.k_scores:
        with st.expander(f"Почему выбрано {analysis.result.n_topics} тем"):
            st.plotly_chart(charts.k_scores_chart(analysis.k_scores, analysis.result.n_topics),
                            use_container_width=True)
            st.caption("Силуэт показывает, насколько ответы ближе к своей теме, чем к соседним. "
                       "Это подсказка: если темы кажутся слишком дробными или слишком общими, задайте число вручную.")

# --- Темы ------------------------------------------------------------------------

with tab_topics:
    st.subheader("Названия тем")
    st.caption("Название по умолчанию — три главных ключевых слова. Переименуйте темы так, как удобно HR. "
               "Если дать двум темам одинаковое название, они объединятся.")
    kw = analysis.result.keywords(8)
    sizes = data[TOPIC_COL].value_counts()
    editor_df = pd.DataFrame({
        "Название": [analysis.names[t] for t in range(analysis.result.n_topics)],
        "Ответов": [int(sizes.get(t, 0)) for t in range(analysis.result.n_topics)],
        "Ключевые слова": [", ".join(w) for w in kw],
    })
    edited = st.data_editor(editor_df, key="names_editor", use_container_width=True, hide_index=True,
                            disabled=["Ответов", "Ключевые слова"],
                            column_config={"Название": st.column_config.TextColumn(width="medium")})
    new_names = dict(enumerate(edited["Название"].astype(str)))
    if new_names != analysis.names:
        apply_names(analysis, new_names)
        data = analysis.data

    st.divider()
    names_sorted = data[NAME_COL].value_counts().index.tolist()
    chosen = st.selectbox("Подробнее о теме", names_sorted)
    left, right = st.columns([1, 1])
    with left:
        topic_ids = [t for t, n in analysis.names.items() if n == chosen]
        words: dict[str, float] = {}
        for t in topic_ids:
            for w, v in analysis.result.topic_words[t]:
                words[w] = words.get(w, 0) + v
        img = charts.wordcloud_image(sorted(words.items(), key=lambda x: -x[1]))
        if img is not None:
            st.image(img, use_container_width=True)
        part = data[data[NAME_COL] == chosen]
        st.markdown(f"**{len(part)}** ответов ({len(part) / len(data):.0%} от всех)")
    with right:
        st.markdown("**Типичные ответы**")
        for text in analysis.examples(chosen, 6):
            st.markdown(f"> {text}")

# --- Срезы -----------------------------------------------------------------------

with tab_segments:
    seg_cols = segment_columns(data, exclude=(text_col,))
    dt_cols = date_columns(data, exclude=(text_col,))
    if not seg_cols and not dt_cols:
        st.info("В файле нет колонок для срезов (отдел, стаж, дата и т.п.).")
    if seg_cols:
        segment = st.selectbox("Срез по колонке", seg_cols)
        strong = notable(lift_by_segment(analysis, segment))
        st.subheader("Что выделяется")
        if len(strong):
            for _, row in strong.iterrows():
                st.markdown(
                    f"- **{segment} = {row['Сегмент']}**: причина «{row['Тема']}» встречается "
                    f"**в {row['Подъём']:.1f} раза чаще** обычного — {row['Доля в сегменте']:.0%} ответов "
                    f"против {row['Доля в среднем']:.0%} в среднем ({row['Ответов']} из {row['Размер сегмента']})."
                )
            st.caption("Показаны только отклонения, которые вряд ли объясняются случайностью "
                       "(стандартизированный остаток ≥ 2, как в критерии хи-квадрат).")
        else:
            st.markdown("Заметных отличий между сегментами нет — причины распределены примерно одинаково.")
        st.subheader(f"Доля причин по колонке «{segment}», %")
        st.plotly_chart(charts.segment_heatmap(share_by_segment(analysis, segment), segment),
                        use_container_width=True)
        st.caption("Каждая строка — 100% ответов этого сегмента. Чем темнее клетка, тем чаще причина.")
    if dt_cols:
        st.divider()
        date_col = st.selectbox("Колонка с датой", dt_cols)
        freq = st.radio("Период", ["Квартал", "Месяц"], horizontal=True)
        tl = timeline(analysis, date_col, "QS" if freq == "Квартал" else "MS")
        st.subheader("Как менялась доля причин, %")
        st.plotly_chart(charts.timeline_small_multiples(tl), use_container_width=True)

# --- Ответы ----------------------------------------------------------------------

with tab_answers:
    f1, f2, f3 = st.columns([2, 1, 1])
    topics_filter = f1.multiselect("Темы", data[NAME_COL].value_counts().index.tolist(), placeholder="Все темы")
    only_low = f2.checkbox("Только с низкой уверенностью")
    query = f3.text_input("Поиск по тексту")
    view = data
    if topics_filter:
        view = view[view[NAME_COL].isin(topics_filter)]
    if only_low:
        view = view[view[CONF_COL] < LOW_CONFIDENCE]
    if query:
        view = view[view[text_col].str.contains(query, case=False, na=False, regex=False)]
    shown = [NAME_COL, CONF_COL, text_col] + [c for c in data.columns
                                              if c not in (NAME_COL, CONF_COL, text_col, TOPIC_COL, "clean_text", "lemmas")]
    st.write(f"Показано ответов: **{len(view)}** из {len(data)}")
    st.dataframe(view[shown], use_container_width=True, hide_index=True,
                 column_config={CONF_COL: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.download_button("Скачать результат (Excel)", to_excel(analysis), file_name="причины_увольнений.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
