# --- build: install dependencies and pre-build the vector index -------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

COPY data/units.json ./data/units.json
# Embeds every ability locally; no API key needed at build time.
RUN SWGOH_DATA_DIR=/app/data .venv/bin/swgoh-agent index build

# --- runtime -----------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 swgoh
WORKDIR /app
COPY --from=build --chown=swgoh:swgoh /app/.venv ./.venv
COPY --from=build --chown=swgoh:swgoh /app/data ./data
ENV PATH="/app/.venv/bin:$PATH" SWGOH_DATA_DIR=/app/data PYTHONUNBUFFERED=1
USER swgoh
EXPOSE 8000
CMD ["swgoh-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
