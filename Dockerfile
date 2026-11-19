# Образ веб-сервиса для Hugging Face Spaces (и для запуска где угодно через Docker).
#   docker build -t exit-interview-topics .
#   docker run -p 8501:8501 exit-interview-topics
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Hugging Face Spaces запускает контейнер от пользователя с id 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# Сначала зависимости: этот слой пересобирается, только если поменялся requirements-app.txt.
ARG TORCH_VERSION=2.15.0
RUN pip install --user torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-app.txt .
RUN pip install --user -r requirements-app.txt

COPY --chown=user . .

# Скачиваем модели и считаем эмбеддинги демо-датасета заранее, чтобы первый
# пользователь не ждал загрузки ~10 ГБ. PRELOAD=0 — пропустить (модели скачаются при первом анализе).
ARG PRELOAD=1
RUN if [ "$PRELOAD" = "1" ]; then python -m src.app.warmup; fi

EXPOSE 8501
# XSRF и CORS отключены: Spaces показывает приложение внутри iframe, и без этого не работает загрузка файлов.
CMD ["streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
