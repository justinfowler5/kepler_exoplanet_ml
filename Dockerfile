# Multi-stage uv build — one image for API and Celery worker.
# Builder tag verified: trixie-based uv images (bookworm tags are stale/404).

FROM ghcr.io/astral-sh/uv:0.12.1-python3.12-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Dependency layer: lockfile + project metadata only
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/samples ./data/samples

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------------------
FROM python:3.12-slim-trixie AS runtime

WORKDIR /app

RUN groupadd --gid 1000 kepler \
    && useradd --uid 1000 --gid kepler --shell /bin/bash --create-home kepler

COPY --from=builder --chown=kepler:kepler /app/.venv /app/.venv
COPY --from=builder --chown=kepler:kepler /app/src /app/src
COPY --from=builder --chown=kepler:kepler /app/data /app/data
COPY --chown=kepler:kepler scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KEPLER_ENV=local

USER kepler

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "kepler_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
