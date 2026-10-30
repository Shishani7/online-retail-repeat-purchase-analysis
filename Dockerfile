FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir uv==0.10.12
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit
COPY data/processed ./data/processed
COPY models ./models
COPY examples ./examples
RUN uv sync --frozen --no-dev && useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
CMD ["uvicorn", "retail_repeat_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]
