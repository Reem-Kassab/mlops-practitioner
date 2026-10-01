FROM python:3.11-slim AS builder

WORKDIR /build

# Upgrade packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy package metadata first.
# This allows Docker to cache dependency installation.
COPY pyproject.toml README.md ./

# Copy the application source
COPY src/ ./src/

# Build/install the Python package into a separate directory.
RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    .


# ============================================================
# Runtime image
# ============================================================

FROM python:3.11-slim AS runtime

# Create a non-root user
RUN useradd \
    --create-home \
    --uid 1000 \
    appuser

WORKDIR /app

# Copy installed Python dependencies/application
COPY --from=builder /install /usr/local

# Copy model artifacts into the image
COPY --chown=appuser:appuser models/ /app/models/

COPY --chown=appuser:appuser configs/ /app/configs/

ENV PRODML_PROJECT_ROOT=/app
# The application runs as a non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]