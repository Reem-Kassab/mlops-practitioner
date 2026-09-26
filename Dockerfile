# Builder Stage
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Final Stage
FROM python:3.11-slim
WORKDIR /app


COPY --from=builder /app/.venv /app/.venv
COPY src /app/src
COPY models /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
# Create a non-root user and switch to it for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "taxi_duration.api.main:app", "--host", "0.0.0.0", "--port", "8000"]