FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache

WORKDIR /app
COPY pyproject.toml README.md ./
COPY campusreg ./campusreg
RUN python -m pip install --no-cache-dir ".[api,embeddings,google]"

RUN mkdir -p /app/data /app/evaluation/results /model-cache
EXPOSE 8000
CMD ["uvicorn", "campusreg.main:app", "--host", "0.0.0.0", "--port", "8000"]
